from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency
from collections import Counter
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
from boruta import BorutaPy
import pandas as pd
import numpy as np
import re
import math

def remove_analysis(df, target, miss_rate,Z=1.96):
    for column in df:
        if len ( df[df[column] == "Blank(s)"] ) / len ( df[column] ) > 0.3:
            sample = list ( df.loc[df[column] == "Blank(s)", target] )
            population = list(df.loc[df[column] != "Blank(s)",target])
            #population = list ( df["SEER cause-specific death classification"] )
            if h_analysis ( sample, population, Z, target ) == True:
                df.drop ( column, inplace=True, axis=1 )
    return df

def h_analysis(sample, population, Z, criteria):
    n = len ( population )
    n1 = len ( sample )
    p = (population.count ( criteria )) / n
    p1 = (sample.count ( criteria )) / n1
    E = Z * math.sqrt ( p * (1 - p) / n )
    if p1 < p - E or p1 > p + E:
        return False
    else:
        return True

def transform_character(df):
    df = df.rename ( columns=lambda x: re.sub ( '[^A-Za-z0-9_]+', '', x ) )
    return df
def inv_trans_character(df,titles):
    dict = {}
    for title in titles:
        if re.sub( '[^A-Za-z0-9_]+', '', title ) in df.columns:
            dict[re.sub( '[^A-Za-z0-9_]+', '', title )] = title
    df.rename(columns=dict)
    return df
def chi_square_importance(X,y):
    bestfeatures = SelectKBest ( score_func=chi2, k=25 )
    fit = bestfeatures.fit ( X, y )
    dfscores = pd.DataFrame ( fit.scores_ )
    dfcolumns = pd.DataFrame ( X.columns )
    # concat two dataframes for better visualization
    featureScores = pd.concat ( [dfcolumns, dfscores], axis=1 )
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    featureScores.set_index ( 'Specs' )
    return featureScores

def tree_importance(X, y):
    dt_model = ExtraTreesClassifier ()
    dt_model.fit ( X, y )
    scores = pd.DataFrame ( dt_model.feature_importances_ )
    dfcolumns = pd.DataFrame ( X.columns )
    score = pd.concat([dfcolumns, scores], axis=1 )
    score.columns = ['Specs', 'Score']  # naming the dataframe columns
    return score


def boruta_importance(X,y):
    # define random forest classifier
    forest = RandomForestClassifier ( n_jobs=-1, class_weight='balanced', max_depth=30 )
    forest.fit ( X, y )
    # define Boruta feature selection method
    feat_selector = BorutaPy ( forest, n_estimators='auto', verbose=2, random_state=1 )
    # find all relevant features
    feat_selector.fit ( X.values, y.values )
    dfcolumns = pd.DataFrame ( X.columns )
    dfRank = pd.DataFrame ( feat_selector.ranking_ )

    feat_Rank = pd.concat ( [dfcolumns, dfRank], axis=1 )
    feat_Rank.columns = ['Specs', 'Score']  # naming the dataframe columns
    return feat_Rank

def intersect_list(list1 , list2):
    s = set(list1).intersection ( set(list2) )
    final_list = list (s)
    return final_list


def cramers_V(var1, var2):
    crosstab = np.array ( pd.crosstab ( var1, var2, rownames=None, colnames=None ) )  # Cross table building
    stat = chi2_contingency ( crosstab )[0]  # Keeping of the test statistic of the Chi2 test
    obs = np.sum ( crosstab )  # Number of observations
    mini = min ( crosstab.shape ) - 1  # Take the minimum value between the columns and the rows of the cross table
    return (stat / (obs * mini))


def conditional_entropy(x, y):
    # entropy of x given y
    y_counter = Counter ( y )
    xy_counter = Counter ( list ( zip ( x, y ) ) )
    total_occurrences = sum ( y_counter.values () )
    entropy = 0
    for xy in xy_counter.keys ():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log ( p_y / p_xy )
    return entropy


def theils_u(x, y):
    s_xy = conditional_entropy ( x, y )
    x_counter = Counter ( x )
    total_occurrences = sum ( x_counter.values () )
    p_x = list ( map ( lambda n: n / total_occurrences, x_counter.values () ) )
    s_x = ss.entropy ( p_x )
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

def heat_map(x,final_list,func):
    rows = []
    for var1 in x[final_list]:
        col = []
        for var2 in x[final_list]:
            correl = func( x[var1], x[var2] )
            col.append ( round ( correl, 2 ) )  # Keeping of the rounded value of the correl V
        rows.append ( col )

    result = np.array ( rows )
    corr = pd.DataFrame ( result, columns=final_list, index=final_list )

    mask = np.zeros_like ( corr, dtype=np.bool )
    mask[np.triu_indices_from ( mask )] = True

    fig, ax = plt.subplots ( figsize=(7, 6) )
    ax = sns.heatmap ( corr, annot=True, ax=ax )
    ax.set_title ( "Theil's U Correlation between Variables" )
    plt.show()
    return corr

def corr_df(x, corr_val, corr_matrix):
    # Creates Correlation Matrix and Instantiates

    iters = range ( len ( corr_matrix.columns ) - 1 )
    drop_cols = []

    # Iterates through Correlation Matrix Table to find correlated columns
    for i in iters:
        for j in range ( i ):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            val = item.values
            if abs ( val ) >= corr_val:
                # Prints the correlated feature set and the corr val
                print ( col.values[0], "|", row.values[0], "|", round ( val[0][0], 2 ) )
                drop_cols.append ( i )


    drops = sorted ( set ( drop_cols ) )[::-1]
    # Drops the correlated columns
    for i in drops:
        col = x.iloc[:, (i + 1):(i + 2)].columns.values
        x = x.drop ( col, axis=1 )
    return x
