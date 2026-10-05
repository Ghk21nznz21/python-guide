'''
Allows:
    odds_as_predictor: Odds (bet365) is a predictor 
    pred_days: future days with no data, meant 
        for future predictions and not training or testing
    metric: model can train on MSE/L2 or the profit of actual betting on games
    
    quality_measuring = columns used only to calculate profit 

'''
from sklearn.model_selection import StratifiedKFold
import csv 
import lightgbm as lgb
import numpy as np
import pandas as pd  
import sklearn.metrics
import toml 
from itertools import product 

from module1_model.scoring import ExpectedProfit
from datetime import date, timedelta

from multiprocessing import Pool

import os 

with open("config.toml", mode="r") as fp:
    config = toml.load(fp)
    config = config['module1_model']


def save_results(params: dict, bet_params: list[dict, ], results: list[list, ], score: float):
    headers = 'Models.csv' not in os.listdir('results/')
    with open('results/Models.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if headers:
            headers = [*list(params.keys()), *list(bet_params[0].keys()), 
                       'Score', 'Eff', 'Profit', 'Avg_Odd', 'GamesBetted']
        params = list(params.values())
        writer.writerow(params)
        for bet_param, result in zip(bet_params, results):
            if result[1] > 0: # positive profit 
                writer.writerow([*params, *list(bet_param.values()), score, *result])

def all_combinations(multiple_params: dict):
    ''' Transform tunning dict into list of tunning combos '''
    parameters_combos = []
    keys = multiple_params.keys()
    vals = multiple_params.values()
    for combo in product(*vals):
        params_combo = dict(zip(keys, combo))
        parameters_combos.append(params_combo)
    return parameters_combos


def create_dataset():
    ''' odds_as_predictor: false if few past odds avaliable'''
    odds_as_predictor = config['odds_as_predictor'] == 'True'
    if odds_as_predictor: #Needs diff datasets, due to filtering groups with few games 
        data = pd.read_csv('data/processed/prepared_for_model.csv')
        
    else:
        data = pd.read_csv('data/processed/prepared_for_model_noOdds.csv')
    
    # Remove Future Games
    print('Size', data.shape[0])
    split_date = (date.today() - timedelta(days=config['pred_days'])).strftime("%Y-%m-%d")
    data = data.loc[data['Date'] <= split_date]
    data.dropna(subset=['Result'], inplace=True)
    data.drop(columns=['Diff_Goals'], axis=1, inplace=True)
    data.reset_index(drop=True, inplace=True)
    target = data.pop('Result')
    print('After Size', data.shape[0])
    
    data['Team1_Formation'] = data['Team1_Formation'].astype('category')
    data['Team2_Formation'] = data['Team2_Formation'].astype('category')
    
    if odds_as_predictor:
        kf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=42)
        datasets = []
        for train_index, valid_index in kf.split(data, target):
            train_data, valid_data = data.iloc[train_index], data.iloc[valid_index]
            train_target, valid_target = target.iloc[train_index], target.iloc[valid_index]
            
            trackers, train_data, valid_data = get_trackers(train_data, valid_data, odds_as_predictor)
            dtrain = lgb.Dataset(train_data, label=train_target)
            dvalid = lgb.Dataset(valid_data, label=valid_target)
            
            datasets.append((dtrain, dvalid, trackers))
    else:
        train = data.loc[data['Odd_X'] == 0]
        test = data.loc[data['Odd_X'] != 0]
        trackers, train, test = get_trackers(train, test, odds_as_predictor)
        datasets.append((train, test, trackers))
    return datasets

def objectiveCV(trial, datasets: list):
    cv_scores = []
    results = []

    # Fixed params
    regulizers = [0, 1e-3, 1e-1, 1e0, 1e1, 4]
    params = {
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 30, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 3, 300),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 1500),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1),
        'min_child_weight': trial.suggest_categorical('min_child_weight', [1e-1, 1, 1e1, 1e2]),
        'subsample': trial.suggest_float('subsample', 0.5, 1),
        'reg_alpha': trial.suggest_categorical('reg_alpha', regulizers),
        'reg_lambda': trial.suggest_categorical('reg_lambda', regulizers),
        'force_col_wise': True,
        #['profit', 'multiclass'] - default for Classifier MultiClass
        'objective': trial.suggest_categorical('objective', ['multiclass']) 

    }
    bet_params = all_combinations(config['betting_params']) 
    categoricals = [f'name:{col}' for col in config['categoricals']]
    
    results = [ [] for _ in range(len(bet_params))]
    for dtrain, dvalid, trackers in datasets:    
        print(dtrain.data.columns) 
        lgbm = lgb.LGBMClassifier(**params, verbose_eval=False, categorical_feature=categoricals)
        early_stopping_callback = lgb.early_stopping(stopping_rounds=30, verbose=True)
        lgbm.fit(
            dtrain.data, dtrain.label,
            eval_set=[(dtrain.data, dtrain.label), (dvalid.data, dvalid.label)],
            eval_metric=params['objective'],  
            callbacks=[early_stopping_callback]
        )

        # score
        preds = lgbm.predict_proba(dvalid.data)
        
        max_tot_profit = 0
        with Pool() as pool:
            pool_args = [(ExpectedProfit(trackers, dvalid.label, preds), bet_param) for bet_param in bet_params]
            profits = pool.map(compute_profit_stats, pool_args)

            for i, profit_stats in enumerate(profits):
                results[i].append(profit_stats)
                tot_profit = profit_stats[1] * profit_stats[-1]
                if tot_profit > max_tot_profit:
                    max_tot_profit = tot_profit
                    
        if params['objective'] == 'multiclass':
            # log_loss allows multi-logloss
            score = sklearn.metrics.log_loss(dvalid.label, preds)
            cv_scores.append(score) 
        else:
            #profit 
            cv_scores.append(max_tot_profit) 
        
        if max_tot_profit < 0:
            # not worth to continue with CV
            return min(cv_scores)
    
    score = np.mean(cv_scores)
    if params['objective'] == 'profit':
        score = -score # minimize 
    
    # Avg Results
    results = [np.asarray(np.matrix(cv_results).mean(axis=0)).ravel() for cv_results in results]
    
    # set adicional info to save 
    trial.set_user_attr('results', results)
    
    # save to csv
    save_results(params, bet_params, results, score)
    return score

def get_trackers(train: pd.DataFrame, test: pd.DataFrame, odds_as_predictor: bool):
    
    # Get Trackes from Test
    trackers = test.loc[:,['Date', 'Team1_Name', 'League_Status', 'Odd_X', 'Odd_V1', 'Odd_V2']]
    remove_to_train = ['Date', 'Team1_Name', 'Country']
    # Remove Cols not meant for training
    if not odds_as_predictor:
        remove_to_train.extend(['Odd_X', 'Odd_V1', 'Odd_V2'])
    train.drop(remove_to_train, axis=1, inplace=True)
    test.drop(remove_to_train, axis=1, inplace=True)
    return trackers, train, test
    

# Profit for each combo (takes a lot of time) 
def compute_profit_stats(args): 
    # Needs to be GLOBAL
    exP, bet_param = args
    return exP.model_quality(bet_param)
