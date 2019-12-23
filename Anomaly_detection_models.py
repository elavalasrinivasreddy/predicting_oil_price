# Import libraries
import numpy as np
import pandas as pd

# Load the data
Total_data = pd.read_excel('Total_data.xlsx')
Total_data.drop('Unnamed: 0',axis=1,inplace=True)
Total_data.shape
Total_data.columns
Total_data['TRR_FI301_PV'].unique()
Total_data

#Aug_data = pd.read_csv('Aug_data.csv',index_col=['bucket'],parse_dates=['bucket'])
#Aug_data.drop('Unnamed: 0',axis=1,inplace=True)
#Aug_data.shape
#
#Sep_data = pd.read_csv('Sep_data.csv',index_col=['bucket'],parse_dates=['bucket'])
#Sep_data.drop('Unnamed: 0',axis=1,inplace=True)
#Sep_data.shape
#
#Oct_data = pd.read_csv('Oct_data.csv',index_col=['bucket'],parse_dates=['bucket'])
#Oct_data.drop('Unnamed: 0',axis=1,inplace=True)
#Oct_data.shape
#
#Nov_data = pd.read_csv('Nov_data.csv',index_col=['bucket'],parse_dates=['bucket'])
#Nov_data.drop('Unnamed: 0',axis=1,inplace=True)
#Nov_data.shape


# Label the Pressure data
def label(series):
    label = []
    for i in series.values:
        if 4.1>i or i<3.9:
            label.append('C')
        elif 4.2>i or i<3.8:
            label.append('B')
        elif 4.3>i or i<3.7:
            label.append('A')
        else:
            label.append('N')
    return pd.Series(label)


Total_label = label(Total_data['TRR_PIC301_PV'])
from collections import Counter
Counter(Total_label)

#Aug_label = label(Aug_data['TRR_PIC301_PV'])
#Aug_label.value_counts()
#Aug_label.shape
#
#Sep_label = label(Sep_data['TRR_PIC301_PV'])
#Sep_label.value_counts()
#Sep_label.shape
#
#Oct_label = label(Oct_data['TRR_PIC301_PV'])
#Oct_label.value_counts()
#Oct_label.shape
#
#Nov_label = label(Nov_data['TRR_PIC301_PV'])
#Nov_label.value_counts()
#Nov_label.shape

#Add the label to dataframe
Total_data.columns
Total_label.columns = 'Pressure_label'
Total_label = Total_label.astype('category')
Total_label.dtype

label_data = Total_data.assign(Pressure_label = Total_label)

# Missing values
label_data.isnull().sum()
# Impute Nan values by Back value
label_data.fillna(method='bfill',axis=0,inplace=True)
label_data.shape

# Finding and impute the outliers
for i in label_data.columns:
    minimum = np.nanpercentile(label_data[i].values,5)
    maximum = np.nanpercentile(label_data[i].values,95)
    for j in range(len(label_data)):
        if label_data[i][j] < minimum:
            label_data[i][j] = minimum
        elif  label_data[i][j] > maximum:
            label_data[i][j] = maximum
        else:
            label_data[i][j] = label_data[i][j]
            

label_data['Pressure_label'].value_counts()
# Target variable is imbalanced
from sklearn.utils import resample
# Separate majority and minority classes
df_N = label_data[label_data['Pressure_label']=='N']
df_C = label_data[label_data['Pressure_label']=='C']
df_B = label_data[label_data['Pressure_label']=='B']
df_A = label_data[label_data['Pressure_label']=='A']

# Upsample minority class
df_N_upsampled = resample(df_N, 
                          replace=True,     # sample with replacement
                          n_samples=361030,    # to match majority class
                          random_state=54) # reproducible results

df_A_upsampled = resample(df_A, 
                          replace=True,     # sample with replacement
                          n_samples=361030,    # to match majority class
                          random_state=54) # reproducible results

df_B_upsampled = resample(df_B, 
                          replace=True,     # sample with replacement
                          n_samples=361030,    # to match majority class
                          random_state=54) # reproducible results


# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_C, df_N_upsampled,df_A_upsampled,df_B_upsampled])
df_upsampled['Pressure_label'].value_counts() # Target variable is balanced



############################################################################################################


# Split the data into train and test data
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=10, test_size=30, random_state=54)
sss.get_n_splits(label_data.drop('Pressure_label',axis=1),label_data['Pressure_label'])

for train_index, test_index in sss.split(label_data.drop('Pressure_label',axis=1),label_data['Pressure_label']):
    
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = label_data.drop('Pressure_label',axis=1)[train_index], label_data.drop('Pressure_label',axis=1)[test_index]
    y_train, y_test = label_data['Pressure_label'][train_index], label_data['Pressure_label'][test_index]
    

from sklearn.model_selection import StratifiedShuffleSplit
X = np.array(label_data.drop('Pressure_label',axis=1))
y = np.array(label_data['Pressure_label'])
cv = StratifiedShuffleSplit(y, test_size=25, random_state=54)

for train_idx, test_idx in cv.split:
    X_train=X[train_idx]
    y_train=y[train_idx]
    

label_data_A = label_data[label_data.Pressure_label=='A']
label_data_A.shape

label_data_B = label_data[label_data.Pressure_label=='B']
label_data_B.shape

label_data_C = label_data[label_data.Pressure_label=='C']
label_data_C.shape

label_data_N = label_data[label_data.Pressure_label=='N']
label_data_N.shape
len(label_data_N)

full_data = pd.concat([label_data_N,label_data_C.iloc[:len(label_data_N),],label_data_B.iloc[:len(label_data_N),],label_data_A.iloc[:len(label_data_N),]],axis=0)
full_data = full_data.reset_index()
full_data.drop('index',axis=1,inplace=True)

#########################################################################################################################################################################
'''                                                 ---KNN---                                   '''
##Outliers
#from scipy import stats
#Z = np.abs(stats.zscore(dataset))
#threshold=3
#print(np.where(Z>3))
#print(Z[189][2])
#df_out = dataset[(Z<3).all(axis=1)]
#dataset.shape
#df_out.shape # 20 rows are removed
#df_out.Type.value_counts()

# Normalize the data
from sklearn.preprocessing import normalize
norm_data = normalize(full_data.iloc[:,0:-1])
y_data = pd.Series(full_data['Pressure_label'])

# metric features
X = norm_data.copy()
X.shape
Y = full_data.iloc[:,-1].values
Y.shape

# Splitting the data into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y.astype(str),stratify=Y,test_size=0.25, random_state=54)

#y_was_categorical.astype(str), stratify=y_was_categorical
# Hyperparameter tunning using GridSearchCV
from sklearn.model_selection import GridSearchCV
# KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# parameters
parameters = {'n_neighbors':[3,5,7,9,11,13,15,17,19,21],'weights': ['uniform', 'distance'],'metric':['euclidean', 'manhattan']}
# Classifier
neigh = KNeighborsClassifier()
clf = GridSearchCV(neigh, parameters, verbose=1, cv=3, n_jobs=-1) #cv=5
clf.fit(X_train, Y_train)
clf.best_params_
clf.best_estimator_
clf.best_score_

# Final model
clf = KNeighborsClassifier(metric='manhattan', n_neighbors= 13, weights= 'distance',p=2)
clf.fit(X_train,Y_train)
# Predicting train set results
Y_pred = clf.predict(X_train)
confusion_matrix(Y_train,Y_pred)
accuracy_score(Y_train, Y_pred)

# Predicting test set results
Y_pred = clf.predict(X_test)
confusion_matrix(Y_test,Y_pred)
accuracy_score(Y_test, Y_pred)

# Another method without Hyperparameter tunning

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
accuracy = []
from sklearn.neighbors import KNeighborsClassifier as KNC
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(X_train,Y_train)
    train_acc = np.mean(neigh.predict(X_train)==Y_train)
    test_acc = np.mean(neigh.predict(X_test)==Y_test)
    accuracy.append([train_acc,test_acc])

import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in accuracy],"bo-")
# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in accuracy],"ro-")
plt.legend(["train","test"])

print(accuracy)

#######################################################################################################################
'''                                       ---SVM---                                    '''
full_data.info()

# Missing data
full_data.isnull().any()
full_data.isnull().sum() # No Nan Values
print((full_data==0).sum())

# Statistical Description
full_data.describe()
# Measures of Dispersion
np.var(full_data)
np.std(full_data)
# Skewness and Kurtosis
from scipy.stats import skew, kurtosis
skew(full_data.drop('Pressure_label', axis=1))
kurtosis(full_data.drop('Pressure_label', axis=1))

# Pairplot
import seaborn as sns
sns.pairplot(full_data)
sns.pairplot(full_data, diag_kind = 'kde')
sns.pairplot(full_data, hue='Pressure_label')

# Heatmap
corr = full_data.corr()
print(corr)
sns.heatmap(corr, annot=True)

full_data.columns
# Boxplot
sns.boxplot(full_data['TRR_FI301_PV'],orient='v').set_title('Boxplot of TRR_FI301_PV')
sns.boxplot(full_data['TRR_FI311_PV'],orient='v').set_title('Boxplot of TRR_FI311_PV')
sns.boxplot(full_data['TRR_FI341_PV'],orient='v').set_title('Boxplot of TRR_FI341_PV')
sns.boxplot(full_data['TRR_PIA341_PV'],orient='v').set_title('Boxplot of TRR_PIA341_PV')
sns.boxplot(full_data['TRR_PIC301_PV'],orient='v').set_title('Boxplot of TRR_PIC301_PV')
sns.boxplot(full_data['TRR_PIC311_PV'],orient='v').set_title('Boxplot of TRR_PIC311_PV')
sns.boxplot(full_data['TRR_TI551_PV'],orient='v').set_title('Boxplot of TRR_TI551_PV')
sns.boxplot(full_data['TRR_TIA204_PV'],orient='v').set_title('Boxplot of TRR_TIA204_PV')
sns.boxplot(full_data['TRR_TIA303_PV'],orient='v').set_title('Boxplot of TRR_TIA303_PV')
sns.boxplot(full_data['TRR_TIA532_PV'],orient='v').set_title('Boxplot of TRR_TIA532_PV')
sns.boxplot(full_data['TRR_TIC301_PV'],orient='v').set_title('Boxplot of TRR_TIC301_PV')
sns.boxplot(full_data['TRR_TIC302_PV'],orient='v').set_title('Boxplot of TRR_TIC302_PV')

# Encoding the label data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
full_data['Pressure_label'] = labelencoder.fit_transform(full_data['Pressure_label'])


# Outliers
from scipy import stats
Z = np.abs(stats.zscore(full_data.iloc[:,0:-1]))
print(np.where(Z>3))
print(Z[1694][2])
df_out = dataset_train[(Z<3).all(axis=1)] # 4062 outliers are removed


full_data['Pressure_label'].value_counts() # Classes are balanced

# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(full_data.drop('Pressure_label',axis=1),full_data['Pressure_label'], test_size=0.25, random_state=54)

# Normalization
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

# Fitting the SVR on dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'linear', degree=1)
regressor.fit(X_train, Y_train)

from sklearn.metrics import r2_score
# Predicting the results on train-set

train_pred = regressor.predict(X_train)
train_error = Y_train - train_pred
train_rmse = np.sqrt(np.mean(train_error**2))
print(train_rmse) # 1.130
# Both .score(), r2_score gives us a accuracy score prediction
print(regressor.score(X_train, Y_train)) # -0.02
print(r2_score(Y_train, train_pred)) # -0.02

# Predicting the results on test-set
test_pred = regressor.predict(X_test)
test_error = Y_test - test_pred
test_rmse = np.sqrt(np.mean(test_error**2))
print(test_rmse) # 1.134
# Both .score(), r2_score gives us a accuracy score prediction
print(regressor.score(X_test, Y_test)) # -0.021
print(r2_score(Y_test, test_pred)) # -0.021

# Kernel = 'rbf'

regressor = SVR(kernel = 'rbf', epsilon=1.0, degree=3)
regressor.fit(X_train, Y_train)

from sklearn.metrics import r2_score
# Predicting the results on train-set
train_pred = regressor.predict(X_train)
train_error = Y_train - train_pred
train_rmse = np.sqrt(np.mean(train_error**2))
print(train_rmse) # 0.922
# Both .score(), r2_score gives us a accuracy score prediction
print(regressor.score(X_train, Y_train)) # 0.317
print(r2_score(Y_train, train_pred)) # 0.317

# Predicting the results on test-set
test_pred = regressor.predict(X_test)
test_error = Y_test - test_pred
test_rmse = np.sqrt(np.mean(test_error**2))
print(test_rmse) # 0.928
# Both .score(), r2_score gives us a accuracy score prediction
print(regressor.score(X_test, Y_test)) # 0.316
print(r2_score(Y_test, test_pred)) # 0.316

#####################################################################################################################################
'''                                    Logistic Regression                                                   '''

# Outliers
# Z-score values
#from scipy import stats
#z = np.abs(stats.zscore(num_data))
#threshold = 3
#print(np.where(z>3))
#num_data.shape # 577 rows
#df_out = num_data[(z<3).all(axis=1)]
#df_out.shape # 37 rows are removed

# creating dummy variables
#dummies = pd.get_dummies(dataset[['sex','child']],drop_first=True)
#
#final_data = pd.concat([dummies, df_out], axis=1)
#final_data.shape
#final_data.isnull().sum()
#final_data.dropna(inplace=True)
#
## Converting affairs class into binary format
#for val in final_data.affair:
#	if val >=1:
#		final_data['affair'].replace(val, 1,inplace=True)
#
#final_data['affair'] = final_data['affair'].astype('category')
#final_data.dtypes
#final_data['affair'].value_counts()

print(full_data)
# Matrics of features
X = full_data.iloc[:,0:12].values
X_col = full_data.drop('Pressure_label', axis=1).columns.tolist()
Y = full_data.iloc[:, -1].values


# Model building 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X,Y)

classifier.coef_ # Coefficients of features
classifier.predict_proba(X) # probability values

Y_pred = classifier.predict(X)
full_data['Pressure_pred'] = Y_pred
Y_prob = pd.DataFrame(classifier.predict_proba(X))
new_df = pd.concat([full_data, Y_prob], axis=1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y,Y_pred) #[423 5/108 4] 
print(cm)
type(Y_pred)
acc = sum(Y==Y_pred)/new_df.shape[0]
print(acc)  # 76%
pd.crosstab(Y_pred, Y)




log_data = label_data.copy()

# Target variable is imbalanced
log_data['Pressure_label'].value_counts()

from sklearn.utils import resample
# Separate majority and minority classes
df_N = log_data[log_data['Pressure_label']=='N']
df_C = log_data[log_data['Pressure_label']=='C']
df_B = log_data[log_data['Pressure_label']=='B']
df_A = log_data[log_data['Pressure_label']=='A']

# Upsample minority class
df_N_upsampled = resample(df_N, 
                          replace=True,     # sample with replacement
                          n_samples=360088,    # to match majority class
                          random_state=54) # reproducible results

df_A_upsampled = resample(df_A, 
                          replace=True,     # sample with replacement
                          n_samples=360088,    # to match majority class
                          random_state=54) # reproducible results

df_B_upsampled = resample(df_B, 
                          replace=True,     # sample with replacement
                          n_samples=360088,    # to match majority class
                          random_state=54) # reproducible results


# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_C, df_N_upsampled,df_A_upsampled,df_B_upsampled])
df_upsampled['Pressure_label'].value_counts()

# Over sampling SMOTE technique
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
os = SMOTE(random_state=0)
os_data_X, os_data_Y = os.fit_sample(X, Y)

os_data_X = pd.DataFrame(data=os_data_X, columns=X_col) #X.columns
os_data_Y = pd.DataFrame(data=os_data_Y, columns=['affair']) # Y.columns

# we can check the numbers of our data
print("length of oversampled data is", len(os_data_X))
print("Number of no-affairs in over sampled data", len(os_data_Y[os_data_Y['affair']==0]))
print('Number of affairs in over sampled data', len(os_data_Y[os_data_Y['affair']==1]))
print("Proportion of no-affairs data in oversampled data is ", len(os_data_Y[os_data_Y['affair']==0])/ len(os_data_X))
print("Proportion of affairs data in oversampled data is ", len(os_data_Y[os_data_Y['affair']==1])/ len(os_data_X))

# Recursive Feature Elimination = RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

rfe = RFE(classifier) # RFE(classifier, n_features=20)
rfe = rfe.fit(os_data_X, os_data_Y)
print(rfe.support_)
print(rfe.ranking_)
print(rfe.n_features_)

print("The Recursive Feature Elimination(RFE) has helped us select the following features:")
# Iterate from array de rfe.support_ and pick columns that are == True
fea_cols = rfe.get_support(1) # Most important features
X_final = os_data_X[os_data_X.columns[fea_cols]] # final features
Y_final = os_data_Y['affair']

# Splitting the data into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_final, test_size=0.25, random_state=0)

# Logistic Regression Model Fitting
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

# Predicting the Trainset Results
classifier.coef_ # coefficients of features 
classifier.predict_proba (X_train) # Probability values 

y_pred_train = classifier.predict(X_train)
y_prob_train = pd.DataFrame(classifier.predict_proba(X_train.iloc[:,:]))

X_train["y_pred"] = y_pred_train
new_df_train = pd.concat([X_train,y_prob_train],axis=1)

from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(Y_train,y_pred_train)
print (cm_train) #[190 129/117 206]
acc_train = sum(Y_train==y_pred_train)/X_train.shape[0]
print(acc_train) # 61% 
pd.crosstab(y_pred_train,Y_train)

# Predicting the test results and calculating the accuracy
Y_pred_test = classifier.predict(X_test)
print('Accuracy of Logistic Regression classifier on test set:{:.2f}'.format(classifier.score(X_test, Y_test)))

# Confusion Matrix
cm_test = confusion_matrix(Y_test,Y_pred_test)
print(cm_test) #[72 37/37 68]
acc_test = sum(Y_test==Y_pred_test)/X_test.shape[0]
print(acc_test) # 65% 

# Compute precision, recall, F-measure and support
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_test))

# ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

class_roc_auc = roc_auc_score(Y_test, Y_pred_test)
print(class_roc_auc)
fpr, tpr, thresholds = roc_curve(Y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)'%class_roc_auc)
plt.plot([0,1], [0,1],'r--');plt.xlim([0.0, 1.0]);plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic');plt.legend(loc="lower right")

# Logistic Regression on total dataset
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression()
classifier1.fit(X_final, Y_final)

classifier1.coef_ # coefficients of features 
classifier1.predict_proba (X_final) # Probability values 

y_pred1 = classifier1.predict(X_final)
os_data_X["y_pred"] = y_pred1
y_prob1 = pd.DataFrame(classifier1.predict_proba(X_final.iloc[:,:]))
new_df1 = pd.concat([os_data_X,y_prob1],axis=1)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(Y_final,y_pred1)
print (cm1) # [266 162/156 272]
type(y_pred1)
acc1 = sum(Y_final==y_pred1)/os_data_X.shape[0]
print(acc1) # 62%
pd.crosstab(y_pred1,Y_final)

# Compute precision, recall, F-measure and support
from sklearn.metrics import classification_report
print(classification_report(Y_final, y_pred1))

# ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

class1_roc_auc = roc_auc_score(Y_final, y_pred1)
print(class1_roc_auc)
fpr, tpr, thresholds = roc_curve(Y_final, classifier1.predict_proba(X_final)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)'%class1_roc_auc)
plt.plot([0,1], [0,1],'r--');plt.xlim([0.0, 1.0]);plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic');plt.legend(loc="lower right")

#################################################################################################################
'''                        XGBOOST Random Forest with bayesian-optimization              '''
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#Bayesian optimization
def bayesian_optimization(dataset, function, parameters):
   X_train, y_train, X_test, y_test = dataset
   n_iterations = 5
   gp_params = {"alpha": 1e-4}

   BO = BayesianOptimization(function, parameters)
   BO.maximize(n_iter=n_iterations, **gp_params)

   return BO.max

bay_opt = bayesian_optimization(label_data,)

def rfc_optimization(cv_splits):
    def function(n_estimators, max_depth, min_samples_split):
        return cross_val_score(
               RandomForestClassifier(
                   n_estimators=int(max(n_estimators,0)),                                                               
                   max_depth=int(max(max_depth,1)),
                   min_samples_split=int(max(min_samples_split,2)), 
                   n_jobs=-1, 
                   random_state=42,   
                   class_weight="balanced"),  
               X=X_train, 
               y=y_train, 
               cv=cv_splits,
               scoring="roc_auc",
               n_jobs=-1).mean()

    parameters = {"n_estimators": (10, 1000),
                  "max_depth": (1, 150),
                  "min_samples_split": (2, 10)}
    
    return function, parameters

'''
params = {'colsample_bynode': 0.8,'learning_rate': 1,'max_depth': 5,
          'num_parallel_tree': 100,'objective': 'binary:logistic','subsample': 0.8,
          'tree_method': 'gpu_hist'}
# Train the model
bst = train(params, dmatrix, num_boost_round=4)
    '''

def xgb_optimization(cv_splits, eval_set):
    def function(eta, gamma, max_depth):
            return cross_val_score(
                   xgb.XGBClassifier(
                       objective="binary:logistic",
                       learning_rate=max(eta, 0),
                       gamma=max(gamma, 0),
                       max_depth=int(max_depth),                                               
                       seed=42,
                       nthread=-1,
                       scale_pos_weight = len(y_train[y_train == 0])/
                                          len(y_train[y_train == 1])),  
                   X=X_train, 
                   y=y_train, 
                   cv=cv_splits,
                   scoring="roc_auc",
                   fit_params={
                        "early_stopping_rounds": 10, 
                        "eval_metric": "auc", 
                        "eval_set": eval_set},
                   n_jobs=-1).mean()

    parameters = {"eta": (0.001, 0.4),
                  "gamma": (0, 20),
                  "max_depth": (1, 2000)}
    
    return function, parameters

#Train model
def train(X_train, y_train, X_test, y_test, function, parameters):
    dataset = (X_train, y_train, X_test, y_test)
    cv_splits = 4
    
    best_solution = bayesian_optimization(dataset, function, parameters)      
    params = best_solution["params"]

    model = RandomForestClassifier(
             n_estimators=int(max(params["n_estimators"], 0)),
             max_depth=int(max(params["max_depth"], 1)),
             min_samples_split=int(max(params["min_samples_split"], 2)), 
             n_jobs=-1, 
             random_state=42,   
             class_weight="balanced")

    model.fit(X_train, y_train)
    
    return model

##########################################################################################################
'''                        Random Forest with Randomized GridSearchCV                         '''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

RSEED = 50

# Load in data
#df = pd.read_csv('https://s3.amazonaws.com/projects-rf/clean_data.csv')

# Full dataset: https://www.kaggle.com/cdc/behavioral-risk-factor-surveillance-system
label_data['Pressure_label'].value_counts()

# Extract the labels
labels = np.array(label_data.pop('Pressure_label'))

# 25% examples in test data
train, test, train_labels, test_labels = train_test_split(label_data,labels, stratify = labels,
                                                          test_size = 0.25, random_state = 54)


# Imputation of missing values
train.isnull().any()
test.isnull().any()
#train = train.fillna(train.mean())
#test = test.fillna(test.mean())

# Features for feature importances
features = list(train.columns)

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               random_state=54, 
                               max_features = 'sqrt',
                               n_jobs=-1, verbose = 1)

# Fit on training data
model.fit(train, train_labels)

n_nodes = []
max_depths = []

# Stats about the trees in random forest
for ind_tree in model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
    
print(f'Average number of nodes {int(np.mean(n_nodes))}') # 844
print(f'Average maximum depth {int(np.mean(max_depths))}') # 19

# Training predictions (to demonstrate overfitting)
train_rf_predictions = model.predict(train)
train_rf_probs = model.predict_proba(train)[:, 1]

# Testing predictions (to determine performance)
rf_predictions = model.predict(test)
rf_probs = model.predict_proba(test)[:, 1]

from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(test,rf_predictions)
print (cm_train) #[190 129/117 206]
acc_train = sum(Y_train==y_pred_train)/X_train.shape[0]
print(acc_train) # 61% 
pd.crosstab(y_pred_train,Y_train)

# Predicting the test results and calculating the accuracy
Y_pred_test = classifier.predict(X_test)
print('Accuracy of Logistic Regression classifier on test set:{:.2f}'.format(classifier.score(X_test, Y_test)))

# Confusion Matrix
cm_test = confusion_matrix(Y_test,Y_pred_test)
print(cm_test) #[72 37/37 68]
acc_test = sum(Y_test==Y_pred_test)/X_test.shape[0]
print(acc_test) # 65% 

# Compute precision, recall, F-measure and support
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_test))

# ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

class_roc_auc = roc_auc_score(Y_test, Y_pred_test)
print(class_roc_auc)
fpr, tpr, thresholds = roc_curve(Y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)'%class_roc_auc)
plt.plot([0,1], [0,1],'r--');plt.xlim([0.0, 1.0]);plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic');plt.legend(loc="lower right")





































from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18

def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, 
                                     [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, 
                                      [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.show();

evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)
plt.savefig('roc_auc_curve.png')

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

# Confusion matrix
cm = confusion_matrix(test_labels, rf_predictions)
plot_confusion_matrix(cm, classes = ['Poor Health', 'Good Health'],
                      title = 'Health Confusion Matrix')

plt.savefig('cm.png')



##################################################################################################################

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb

# Encoding the label data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data = label_data.copy()
data['Pressure_label'] = labelencoder.fit_transform(data['Pressure_label'])

#columns = data.drop(['Pressure_label'], axis=1).columns
#for column in columns:
#    uniqueVals = data[column].unique()
#    if not isinstance(uniqueVals[0], int):
#        mapper = dict(zip(uniqueVals, range(len(uniqueVals))))
#        label_data[column] = label_data[column].map(mapper).astype(int)
#        test[column] = test[column].map(mapper).astype(int)

X = data.drop('Pressure_label',axis=1).values
Y = data['Pressure_label'].values

# Splite the data into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=54)

# print("Train a Gradient Boosting model")
# clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.005, subsample=0.7,
#                                      min_samples_leaf=10, max_depth=7, random_state=11)
print("Train a Random Forest model")
clf = RandomForestClassifier(n_estimators=25)

clf.fit(X_train, Y_train)

print("Train a XGBoost model")
params = {"objective": "reg:linear",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=250
gbm = xgb.train(params, xgb.DMatrix(X_train, Y_train), num_trees)
gbm_pred = gbm.predict(xgb.DMatrix(X_test))
clf_pred = clf.predict_proba(X_test)

print("Make predictions on the test set")
test_probs = (clf_pred + gbm_pred)/2

result = pd.DataFrame({'id': test.index})
result['Hazard'] = clf.predict_proba(test[columns])[:, 1]
result.to_csv('result.csv', index=False, sep=',')









