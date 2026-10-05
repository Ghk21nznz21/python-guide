import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


def train_models(X, y, model, param_grid, cv, scoring):
    """
    Train a model with hyperparameter tuning using GridSearchCV and print the best model's parameters and feature weights (if applicable).

    @param X: Features
    @param y: Target
    @param model: Estimator object
    @param param_grid: Dictionary of hyperparameter values to search
    @param cv: Cross-validation strategy
    @param scoring: Scoring metric for optimization

    @return: Best trained model
    """
    grid_search = RandomizedSearchCV(model, param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    print("Best Score:", best_score)
    print("Best Parameters:", best_params, '\n \n')

    ## Some model types have this attr 
    #if hasattr(best_model, 'coef_'): # It could be a type of model that dosent have coef__
    #    print("Feature Weights (Coefficients):")
    #    coef_names = X.columns if hasattr(X, 'columns') else range(X.shape[1])
    #    coef_values = best_model.coef_
    #    sorted_indices = np.argsort(-np.abs(coef_values))
    #    sorted_names = [coef_names[i] for i in sorted_indices]
    #    sorted_coefs = [coef_values[i] for i in sorted_indices]
    #    for name, coef in zip(sorted_names, sorted_coefs):
    #        print(f"\t{name}: {round(coef, 4)}")
    #else:
    #    print("Model does not have coefficients.")
    return best_model


def mesh_grid3D(grid_search, param_grid):
    """
    Create a 3D or 2D mesh grid plot to visualize hyperparameter tuning results.

    @param grid_search: GridSearchCV object
    @param param_grid: Dictionary of hyperparameter values to search

    Note: This function requires the `matplotlib` library for visualization.
    """
    # Extract the mean test scores and hyperparameters from the cv_results_
    mean_test_scores = grid_search.cv_results_['mean_test_score']
    params = grid_search.cv_results_['params']

    # Get the keys and values of the parameter grid
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())

    if len(param_keys) > 1:
        # Generate a meshgrid for plotting in 3D
        param_grid_mesh = np.meshgrid(*param_values, indexing='ij')

        # Reshape the mean test scores to match the grid shape
        mean_test_scores_grid = np.array(mean_test_scores).reshape(param_grid_mesh[0].shape)

        # Plot the results in 3D
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(param_grid_mesh[0], param_grid_mesh[1], mean_test_scores_grid, c=mean_test_scores_grid, cmap='viridis')

        # Set labels for each parameter
        for i in range(len(param_keys)):
            ax.set_xlabel(param_keys[i])

        ax.set_zlabel('Mean Test Score')
    else:
        # Plot the results in 2D for a single parameter
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.plot(param_values[0], mean_test_scores, marker='o')

        # Set labels for the single parameter
        ax.set_xlabel(param_keys[0])
        ax.set_ylabel('Mean Test Score')

    plt.title('Grid Search Results')
    plt.show()
    
    
# FAKE DATA to Test

# Create a fake dataset
X, y = make_classification(n_samples=100, n_features=20, random_state=42)

# Define a model
model = LogisticRegression()

# Define a parameter grid for hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}

# Define cross-validation strategy
cv = 5

# Define scoring metric for optimization
scoring = 'accuracy'

# Train the model
best_model = train_models(X, y, model, param_grid, cv, scoring)

# Visualize the results
mesh_grid3D(best_model, param_grid)