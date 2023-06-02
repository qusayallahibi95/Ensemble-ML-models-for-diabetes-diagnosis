# Recursive Feature Elimination RFE
#1 pandad libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# read the data set
ds = pd.read_csv('Diabetes_DS.csv')

# mark the 0 as nan and fill it by using mean
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
for column in zero_not_accepted:
    ds[column] = ds[column].replace(0,np.NaN)
    #mean = int (ds[column].mean(skipna=True))
    #ds[column] = ds[column].replace(np.NaN,mean)

ds.loc[(ds['Outcome'] == 0 ) & (ds['Insulin'].isnull()), 'Insulin'] = 102.5
ds.loc[(ds['Outcome'] == 1 ) & (ds['Insulin'].isnull()), 'Insulin'] = 169.5

ds.loc[(ds['Outcome'] == 0 ) & (ds['Glucose'].isnull()), 'Glucose'] = 107
ds.loc[(ds['Outcome'] == 1 ) & (ds['Glucose'].isnull()), 'Glucose'] = 140

ds.loc[(ds['Outcome'] == 0 ) & (ds['SkinThickness'].isnull()), 'SkinThickness'] = 27
ds.loc[(ds['Outcome'] == 1 ) & (ds['SkinThickness'].isnull()), 'SkinThickness'] = 32

ds.loc[(ds['Outcome'] == 0 ) & (ds['BloodPressure'].isnull()), 'BloodPressure'] = 70
ds.loc[(ds['Outcome'] == 1 ) & (ds['BloodPressure'].isnull()), 'BloodPressure'] = 74.5

ds.loc[(ds['Outcome'] == 0 ) & (ds['BMI'].isnull()), 'BMI'] = 30.1
ds.loc[(ds['Outcome'] == 1 ) & (ds['BMI'].isnull()), 'BMI'] = 34.3


#split the target
target_name ='Outcome'
y =ds[target_name]
X =ds.drop([target_name], axis=1)



from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC , LinearSVC
from sklearn.ensemble import RandomForestClassifier

for num_feats in range(1,2):
    rfe_selector = RFE(estimator=DecisionTreeClassifier(), importance_getter= "auto", n_features_to_select= None)
    rfe_selector.fit(X, y)
    rfe_support = rfe_selector.get_support()
    #importance =rfe_selector.importance_getter
    #params = rfe_selector.get_params()
    
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    #print(str(len(rfe_feature)), 'selected features')
    print('Selected features= ',rfe_feature)
    
# summarize all features
for i in range(X.shape[1]):
 print('Column: %d, Selected %s,Rank: %.3f' % (i, rfe_selector.support_[i], rfe_selector.ranking_[i]))


"""
# show the importance feacure
name =list (train.columns.values)
importance =model.feature_importances_
for i in range (train.shape[1]):
    print(name[i],importance[i]*100)
tree.plot_tree(model)
"""