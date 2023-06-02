# ALL Techniques
#libraries
#1 pandad libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from numpy import mean
from numpy import std
import seaborn as sns

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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
#
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import StackingClassifier

#confusion && accuracy &&classification_report libraries
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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



#RFE  DT
data_sample = data_sample[['Glucose', 'SkinThickness', 'Insulin', 'Age']]


#Normalization
x_scaler = MinMaxScaler()
x_scaler.fit(data_sample)
column_names = data_sample.columns
data_sample[column_names] = x_scaler.transform(data_sample)

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score),
           'roc_auc' : make_scorer(roc_auc_score)}


seed = 42
kfold = model_selection.KFold(n_splits=5, random_state=seed, shuffle= True)


print("**************************************************\n")

knn = KNeighborsClassifier(n_neighbors =4, weights= 'distance', algorithm= 'ball_tree', leaf_size = 40, metric= 'minkowski')
# evaluate model
scores = cross_val_score(knn, data_sample, data_target_sample, scoring='accuracy', cv=kfold, n_jobs=-1)
# report performance
print('Accuracy of knn = %.3f' % (mean(scores*100)))

results = model_selection.cross_val_predict(estimator=knn,
                                          X=data_sample,
                                          y=data_target_sample,
                                          cv=kfold,)

#Print confusion matrix 
conf_matrix = confusion_matrix(y_true=data_target_sample, y_pred=results)
plt.figure(figsize=(6,6))
plt.title('Confusion matrix of knn Technique')
sns.heatmap(conf_matrix,annot=True, fmt=".1f")
plt.xlabel('Predicted')
plt.xticks(rotation=45)
plt.ylabel('True')
plt.ioff()
plt.show()
plt.show()


TN, FP, FN, TP = confusion_matrix(y_true=data_target_sample, y_pred=results).ravel()
print("confusion matrix values =\n",TN, FP, FN, TP)

# 1-accuracy
accuracy =(TP+TN)/(TP+TN+FP+FN)

# 2-Precision or positive predictive value
Precision = TP/(TP+FP)

# 3-Sensitivity, hit rate, recall, or true positive rate
Sensitivity = TP/(TP+FN)

# 4-Specificity or true negative rate
Specificity = TN/(TN+FP) 

# 5-F1 score 2TP⁄(2TP+FP+FN)
F1score =(2*TP)/(2*TP+FP+FN)
# or 
#F1score =2*(Precision*Sensitivity/(Precision+Sensitivity))

print("Accuracy =", accuracy*100)
print("Precision = ",Precision*100)
print("Recall = ",Sensitivity*100)
print("F1 score = ",F1score*100)