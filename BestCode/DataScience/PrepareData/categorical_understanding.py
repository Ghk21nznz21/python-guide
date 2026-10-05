import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CategoricalAnalyses:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.originial_df = self.df.copy()
        
    def steps(self):
        '''
        pd.Describe
        sns.countplot
        pd.Boxplots
        '''
        pass

    def countplot(self, column: str, **kwargs):
        ''' Show the distribution of a feature 
        Use kwargs to set the plot properties 
        '''
        self.df[column].value_counts().sort_index().plot.bar(**kwargs)
        plt.title('Original Dataset')
        plt.xlabel(column.replace('_', ' '))
        plt.ylabel('Count')
        plt.show()
        
    def countplot_sns(self, cat_ord_vars):
        fig, axes = plt.subplots(
            nrows=len(cat_ord_vars), ncols=1,
            figsize=(10, len(cat_ord_vars) * 4))
        for i, var in enumerate(cat_ord_vars):
            sns.countplot(x=var, data=self.df, ax=axes[i])
            axes[i].set_title(f'Distribution of {var}', fontsize=14, fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45, labelsize=10)
        plt.tight_layout()
        plt.show()


    def plot_groupedbar(self, feature: str, target: str, 
                         title: str, feat_by_targ: bool):
      '''
      Shows the distribution of a feature conditioned to the Target 
      (Relative and Absolute frequencies)
        - Feature x Target
        - Target x Feature
      '''
      feature_list = sorted(list(self.df[feature].unique()))
      target_list = sorted(list(self.df[target].unique()))

      new_feat_list = list()
      new_targ_list = list()
      new_count_list = list()

      for k, feat in enumerate(feature_list):
        for i, targ in enumerate(target_list):
          count = self.df.loc[
             (self.df[feature] == feat) \
              & (self.df[target] == targ), target].count()

          new_feat_list.append(feat)
          new_targ_list.append(targ)
          new_count_list.append(count)

      df_plot = pd.DataFrame({
          'Feature': new_feat_list,
          'Target': new_targ_list,
          'Count': new_count_list
      })

      # Set the desired figure size
      plt.figure(figsize=(10, 6))

      totals = list()
      if feat_by_targ:
        for i in list(feature_list):
          totals.append(df_plot[df_plot["Feature"].isin([i])].Count.sum())
        totals.extend(totals)

        # Plot the grouped bar chart using Seaborn
        ax = sns.barplot(x='Feature', y='Count', hue='Target', data=df_plot, 
                         palette=sns.color_palette("pastel", as_cmap=False))
      else:
        for i in new_targ_list:
          totals.append(df_plot[df_plot["Target"].isin([i])].Count.sum())

        # Plot the grouped bar chart using Seaborn
        ax = sns.barplot(x='Target', y='Count', hue='Feature', data=df_plot, 
                         palette=sns.color_palette("pastel", as_cmap=False))

      ax.set(title=title, ylim=(0, df_plot['Count'].max() + 50000))
      plt.xlabel('')
      plt.ylabel('')
      plt.legend(loc='upper right', ncols=2)

      # Add value labels to the bars
      for k, p in enumerate(ax.patches):
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        value = int(y)
        percentage = f'{(y / totals[k]) * 100:.0f}%'
        ax.annotate(f'{value:,}', (x, y), ha='center', va='bottom')
        ax.annotate(percentage, (x, y-(0.5*y)), ha='center', va='center')
        
      plt.show()


    def filter_col_value(self, column: str, value: 'value', 
                         group_names=None, cols_wanted=None, 
                        include_ocurrences=False):
        ''' 
        Filter by col and values on that col 
        Allow grouping 
        Allow filtering by cols
        '''
        filtered = self.df[self.df[column] == value]
        if type(group_names) == list:
            filtered = filtered.groupby(by=group_names)
            if type(group_names) == list:
                filtered = filtered.loc[:, cols_wanted]
            if include_ocurrences:
                filtered = filtered.count().to_frame().\
                    rename(columns={'id': 'count'}).reset_index()
        return filtered


