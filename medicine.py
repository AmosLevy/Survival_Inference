import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
import lightgbm as lgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import optuna
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix
import shap
from functions import *

#optional, feature selection from csv
ft = pd.read_csv ( 'features.csv', low_memory=False )
print ( ft.shape )
features = list ( ft.columns )

df = pd.read_csv ( 'seer.csv', low_memory=False )
print ( df.shape )

df = df[features]
print ( df.shape )

#remove rows with no target value
df = df[df["SEER cause-specific death classification"] != "Dead (missing/unknown COD)"]
print ( df.shape )

#remove rows before 2010 and after 2014
df = df[(df["Year of diagnosis"] > 2010) & (df["Year of diagnosis"] < 2014) ]
print ( df.shape )

# change all NA to "Blank(s)" (better to recognize missing values)
df = df.fillna ( "Blank(s)" )

# Remove all columns with unique value
for col in df.columns:
    if len ( df[col].unique () ) == 1:
        df.drop ( col, inplace=True, axis=1 )
print ( df.shape )

# Hypotethis analysis to remove columns with a lot of missing value and don't have influence on target data
df = remove_analysis(df,"SEER cause-specific death classification",miss_rate = 0.3)
print ( df.shape )
print(df["SEER cause-specific death classification"].value_counts())

df.info()
coded_df = df.copy()

#separate X and y and encode y > 0=dead , 1 =alive
y = coded_df["SEER cause-specific death classification"]
X = coded_df.drop ( ["SEER cause-specific death classification"], axis=1 )
y =y.replace("Dead (attributable to this cancer dx)",  0)
y =y.replace("Alive or dead of other cause",  1)

#split data
Xtrain, Xtest, yc, ytest = train_test_split (X, y, train_size=0.8, test_size=0.2, random_state=1,stratify=y )

# encode categorical Data
Xc = Xtrain.copy()
ordinal_encoder = OrdinalEncoder()
notnum_col = [cname for cname in Xtrain.columns if Xtrain[cname].dtype not in ['int64', 'float64']]
Xc[notnum_col] = ordinal_encoder.fit_transform(Xtrain[notnum_col])

# chi square importance
chi_importance = chi_square_importance(Xc,yc)
chi_importance.to_csv("chi square.csv")
#chi_importance.nsmallest ( 10, 'Score', keep='all' ).plot.barh ( x='Specs', y='Score' )

# DT importance
dt_importance = tree_importance(Xc,yc)
dt_importance.to_csv("DT.csv")
#dt_importance.nlargest ( 10, 'Score', keep='all' ).plot.barh ( x='Specs', y='Score' )

# Boruta
boruta_imp = boruta_importance(Xc,yc)
boruta_imp.to_csv("boruta.csv")
#boruta_imp .nsmallest ( 10, 'Score', keep='all' ).plot.barh ( x='Specs', y='Score' )

# create lists of features
Chi2_features = list ( chi_importance.nlargest ( 30, 'Score', keep='all' )['Specs'] )
DT_features = list ( dt_importance.nlargest ( 30, 'Score', keep='all' )['Specs'] )
Boruta_features = list (boruta_imp[boruta_imp['Score']<11]['Specs'] )

# Calculates intersection of methods
#final_list = intersect_list(Chi2_features,Boruta_features)
final_list = Boruta_features
# heat map cramer
#corr = heat_map(Xc,final_list,cramers_V)

# heat map Theil's U
corr = heat_map(Xc,final_list,theils_u)

#Removed corraleted features
final = corr_df( Xtrain[final_list], 0.8, corr )
f_list = final.columns
print(f_list)

Xtest = Xtest[f_list]
Xtrain = Xtrain[f_list]

# Running all the models
clf = LazyClassifier ( verbose=0, ignore_warnings=True )
models, predictions = clf.fit ( Xtrain, Xtest, yc, ytest )
models.to_csv ( "models.csv", encoding='utf-8', index=True )

#Run LGBM
for c in Xtrain.columns:
    col_type = Xtrain[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        Xtrain[c] = Xtrain[c].astype('category')

for c in Xtest.columns:
    col_type = Xtest[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        Xtest[c] = Xtest[c].astype('category')

lgbmodel = lgb.LGBMClassifier()
X_train =transform_character(Xtrain)
X_test = transform_character(Xtest)
lgbmodel.fit(X_train, yc)
y_pred = lgbmodel.predict ( X_test )

accuracy = accuracy_score ( y_pred, ytest )
print ( 'LightGBM Model accuracy score: {0:0.4f}'.format ( accuracy_score ( ytest, y_pred ) ) )
print ( 'LightGBM Model Balanced accuracy score: {0:0.4f}'.format ( balanced_accuracy_score ( ytest, y_pred ) ) )

# hyperparameters optimization
def objective(trial):
    train_x, test_x, train_y, test_y = train_test_split ( X_train, yc , test_size=0.3 )
    dtrain = lgb.Dataset ( train_x, label=train_y )

    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'lambda_l1': trial.suggest_loguniform ( 'lambda_l1', 1e-8, 10.0 ),
        'lambda_l2': trial.suggest_loguniform ( 'lambda_l2', 1e-8, 10.0 ),
        'num_leaves': trial.suggest_int ( 'num_leaves', 2, 256 ),
        'feature_fraction': trial.suggest_uniform ( 'feature_fraction', 0.4, 1.0 ),
        'bagging_fraction': trial.suggest_uniform ( 'bagging_fraction', 0.4, 1.0 ),
        'bagging_freq': trial.suggest_int ( 'bagging_freq', 1, 7 ),
        'min_child_samples': trial.suggest_int ( 'min_child_samples', 5, 100 ),
    }

    gbm = lgb.train ( param, dtrain )
    preds = gbm.predict ( test_x )
    pred_labels = np.rint ( preds )
    accuracy = sklearn.metrics.balanced_accuracy_score( test_y, pred_labels )
    return accuracy

# study = optuna.create_study(sampler = CmaEsSampler(),direction='maximize')
study = optuna.create_study(direction='maximize' )
study.optimize(objective, n_trials=150 )

parameter = study.best_trial.params

lgbmodel = lgb.LGBMClassifier ( **parameter )
lgbmodel.fit ( X_train, yc )
y_pred = lgbmodel.predict ( X_test )


accuracy = accuracy_score ( y_pred, ytest )
print ( 'LightGBM Model accuracy score: {0:0.4f}'.format ( accuracy_score ( ytest, y_pred ) ) )
print ( 'LightGBM Model Balanced accuracy score: {0:0.4f}'.format ( balanced_accuracy_score ( ytest, y_pred ) ) )

# Confusion Matrix
cf_matrix = confusion_matrix(ytest, y_pred)
group_names = ['True Negative','False Positive','False Negative','True Positive']
group_counts = ['{0:0.0f}'.format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.show()

# Result explanation by Shapley values
explainer = shap.TreeExplainer(lgbmodel)
shap_values = explainer.shap_values(Xtrain)
shap.summary_plot(shap_values, Xtrain)
shap.summary_plot(shap_values[1], Xtrain)
shap.initjs ()
explainer = shap.TreeExplainer(lgbmodel,feature_perturbation="tree_path_dependent")
#shap.decision_plot(explainer(Xtrain),Xtrain.iloc[0,:])
#shap.force_plot(explainer.expected_value, shap_values.values[0, :], Xtrain.iloc[0, :])
shap.plots.waterfall(shap_values[0])
plt.show()