#libraries
#1 pandad libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from numpy import mean
from numpy import std

# ANOVA feature selection for numeric input and categorical output
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
#preprocessing and split libraries
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

#models libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC , LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
#
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import StackingClassifier

#confusion && accuracy &&classification_report libraries
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics

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
data_target =ds[target_name]
data =ds.drop([target_name], axis=1)


#
ds_0 = ds[ds['Outcome']==0]
ds_1 = ds[ds['Outcome']==1]

q1 = ds_0['Pregnancies'].quantile(0.25)
q3 = ds_0['Pregnancies'].quantile(0.75)
IQR = q3 - q1
upper_whisker = q3+1.5*IQR
lower_whisker = q1-1.5*IQR
print(upper_whisker, lower_whisker)

ds['Pregnancies'] = np.where((ds['Outcome']==0) & (ds['Pregnancies']>upper_whisker), upper_whisker, ds['Pregnancies'])
ds['Pregnancies'] = np.where((ds['Outcome']==0) & (ds['Pregnancies']<lower_whisker), lower_whisker, ds['Pregnancies'])

for i in data.columns:
    if i!='Outcome' and i!='Pregnancies':
        q1 = ds_0[i].quantile(0.25)
        q3 = ds_0[i].quantile(0.75)
        iqr = q3 - q1
        upper_whisker = q3+1.5*iqr
        lower_whisker = q1-1.5*iqr
        data[i] = np.where((ds['Outcome']==0) & (ds[i]>upper_whisker), upper_whisker, ds[i])
        data[i] = np.where((ds['Outcome']==0) & (ds[i]<lower_whisker), lower_whisker, ds[i])
        
        q1 = ds_1[i].quantile(0.25)
        q3 = ds_1[i].quantile(0.75)
        iqr = q3 - q1
        upper_whisker = q3+1.5*iqr
        lower_whisker = q1-1.5*iqr
        data[i] = np.where((ds['Outcome']==1) & (ds[i]>upper_whisker), upper_whisker, ds[i])
        data[i] = np.where((ds['Outcome']==1) & (ds[i]<lower_whisker), lower_whisker, ds[i])



smote = SMOTE()
data_sample, data_target_sample = smote.fit_resample(data, data_target)

print('Original dataset \n',data_target.value_counts()) 
print('Resample dataset \n', data_target_sample.value_counts())

# pearson correlation featuer slection
#data_sample = data_sample[['Pregnancies', 'Age', 'SkinThickness', 'BMI', 'Insulin', 'Glucose']]
#data_sample = data_sample[['Age', 'SkinThickness', 'BMI', 'Insulin', 'Glucose']]
#data_sample = data_sample[['Age', 'SkinThickness', 'BMI', 'Insulin', 'Glucose']]

#RFE  RF
data_sample = data_sample[['Glucose', 'SkinThickness', 'Insulin', 'Age']]


#Normalization
x_scaler = MinMaxScaler()
x_scaler.fit(data_sample)
column_names = data_sample.columns
data_sample[column_names] = x_scaler.transform(data_sample)





from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score),
           'roc_auc' : make_scorer(roc_auc_score)}



seed = 42
kfold = model_selection.KFold(n_splits=5, random_state=seed, shuffle= True)

#knn = KNeighborsClassifier()
DT = DecisionTreeClassifier()
#SVM =SVC()
#MLP =MLPClassifier()
#RF = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV
#KNN
#params= {'n_neighbors':np.arange(1,20),'weights':['uniform','distance'], 'algorithm':['auto','ball_tree', 'kd_tree','brute'], 'leaf_size':np.arange(20,50)}
#DT
params= {'max_depth':np.arange(1,20),'criterion':['gini','entropy'], 'splitter':['best','random'], 'max_features':['auto','sqrt', 'log2']}
#SVM
#params= {'degree':np.arange(1,10),'gamma':['scale','auto'], 'kernel':['linear','poly', 'rbf','sigmoid']}
#MLP
#params= {'hidden_layer_sizes':np.arange(100,400),'activation':['identity','logistic', 'tanh','relu'], 'solver':['lbfgs','sgd', 'adam'], 'learning_rate':['constant','invscaling', 'adaptive'],'max_iter':np.arange(200,300), 'early_stopping':['True','False'], 'momentum':['0.9','0.8', '0.7','0.6']}

params= {'n_estimators':np.arange(10,30), 'criterion':['friedman_mse','squared_error', 'mse'], 'max_depth':np.arange(3,15)}

GS= GridSearchCV(DT,params,cv=5,scoring='roc_auc')
GS.fit(data_sample,data_target_sample)
print(GS.best_params_)


"""
results = model_selection.cross_validate(estimator=DT,
                                          X=data_sample,
                                          y=data_target_sample,
                                          cv=kfold,
                                          scoring=scoring)

print('test_accuracy', np.mean(results['test_accuracy']*100))
print('test_precision', np.mean(results['test_precision']*100))
print('test_recall',np.mean(results['test_recall']*100))
print('test_f1_score',np.mean(results['test_f1_score']*100))
print('test_roc_auc',np.mean(results['test_roc_auc']*100))

"""