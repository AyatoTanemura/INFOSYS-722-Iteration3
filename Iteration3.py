# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:44:26 2022

@author: OEM
"""

# Iteration3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler

# Importing library to split the data into training part and testing part.
from sklearn.model_selection import train_test_split

# Correlation finding
from sklearn.feature_selection import chi2
import scipy.stats as stats

# Constant feature checking
from sklearn.feature_selection import VarianceThreshold

# RandomOverSampler to handle imbalanced data
from imblearn.over_sampling import RandomOverSampler

# Cross Validation
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC                            
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import zscore

from sklearn.metrics import classification_report
from sklearn import metrics

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
pd.set_option('display.max_columns',29)

# MySQL connection ----------------------------------------------------------------------------------------------------------------------------
# insert MySQL Database information here
import mysql.connector as mysql
from tabulate import tabulate

HOST = "127.0.0.1"
DATABASE = "infosys722"
USER = "root"
PASSWORD = "123456"

try:
    mydb = mysql.connect(host=HOST, database = DATABASE ,user=USER, 
                              passwd=PASSWORD,use_pure=True)
    query = "Select * from general_data;"
    df_sql = pd.read_sql(query,mydb)
    mydb.close() #close the connection
except Exception as e:
    mydb.close()
    print(str(e))

# create dataframe from MySQL database
# https://medium.com/analytics-vidhya/importing-data-from-a-mysql-database-into-pandas-data-frame-a06e392d27d7

# Data Import ------------------------------------------------------------------------------------------------------------------------------------

# Data load
df = pd.read_csv("Data/general_data.csv")
df.shape

# Data check
df.head()
df.tail()

print("Shape of the dataset:",df.shape,'\n')
print("Total number of rows (excluding column name):",df.shape[0],'\n')
print("Total number of column:",df.shape[1],'\n')
print("Columns name: ", df.columns)

# Data Integration
# splitting dataframe by row index
from sqlalchemy import create_engine
df_integ1 = df[['EmployeeID', 'Age', 'Attrition', 'BusinessTravel', 'Department',
       'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
       'Gender', 'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome',
       'NumCompaniesWorked', 'Over18', 'PercentSalaryHike', 'StandardHours',
       'StockOptionLevel']].copy()

df_integ2 = df[['EmployeeID', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
       'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance',
       'JobInvolvement', 'PerformanceRating']].copy()

hostname = "127.0.0.1"
dbname = "infosys722"
uname = "root"
pwd = "123456"

# Create SQLAlchemy engine to connect to MySQL Database
engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
				.format(host=hostname, db=dbname, user=uname, pw=pwd))

# Convert dataframe to sql table                                   
df_integ1.to_sql('df_integ1', engine, index=False)
# Convert dataframe to sql table                                   
df_integ2.to_sql('df_integ2', engine, index=False)

engine.dispose()

# Join two table in python using Mysql
try:
    mydb = mysql.connect(host=hostname, database = dbname ,user=uname, 
                              passwd=pwd,use_pure=True)
    query = "Select * from df_integ1 Join df_integ2 on df_integ1.EmployeeID = df_integ2.EmployeeID;"
    result_dataFrame = pd.read_sql(query,mydb)
    mydb.close() #close the connection
except Exception as e:
    mydb.close()
    print(str(e))
    
result_dataFrame.head()
result_dataFrame.shape
result_dataFrame.columns

    
# Data Quality ---------------------------------------------------------------------------------------------------------------------------------

# Nu8ll values, Data Type handling and checking zeros
print(df.info(),"\n\n")
print("Checking null values: \n")
print(df.isnull().sum())

# Replacing Null values
df.fillna(method = 'ffill', inplace=True)
print(df.isnull().sum())

df.info()

# Handing DataType
df.NumCompaniesWorked = df.NumCompaniesWorked.astype("int64")
df.TotalWorkingYears = df.TotalWorkingYears.astype("int64")
df.EnvironmentSatisfaction = df.EnvironmentSatisfaction.astype("int64")
df.JobSatisfaction = df.JobSatisfaction.astype("int64")
df.WorkLifeBalance = df.WorkLifeBalance.astype("int64")

df.dtypes

# Cheking Zeros
for i in df.columns:
    print(i,len(df[df[i] == 0]))
    
# Outliers
q_50 = df.TotalWorkingYears.quantile(0.5)
q_50

q = df.TotalWorkingYears.quantile(0.95)

df_NoOut1 = df.query('TotalWorkingYears < @q')
df_NoOut1.shape

q_age = df_NoOut1.Age.quantile(0.95)
df_NoOut2 = df_NoOut1.query('Age < @q_age')
df_NoOut2.shape
    
# Data Exploration -----------------------------------------------------------------------------------------------------------------------------
    
# Data Explorations
print(sorted(df.Age.unique()),"\n")
x = sorted(df.Age.unique())
print("Age group who are working in the company:", x[0], "-", x[-1])

# Exploring Categorical Features
cat = ['Attrition','BusinessTravel','Department',
       'Education','EducationField','Gender', 'NumCompaniesWorked',
       'JobLevel','JobRole','MaritalStatus','NumCompaniesWorked',
       'StockOptionLevel','TrainingTimesLastYear','EnvironmentSatisfaction',
       'JobSatisfaction','WorkLifeBalance','JobInvolvement','PerformanceRating']

col=['Age','Attrition','BusinessTravel','Department',
     'Education','EducationField','Gender', 'StandardHours', 
     'NumCompaniesWorked', 'JobLevel','JobRole','MaritalStatus',
     'NumCompaniesWorked','Over18','StockOptionLevel','TrainingTimesLastYear',
     'EnvironmentSatisfaction','JobSatisfaction','WorkLifeBalance','JobInvolvement',
     'PerformanceRating']

for x in col:
    print('{}: {}'.format(x.upper(),sorted(df[x].unique())), "\n")
    
#Count of each category
col1 = ['Age','Attrition','BusinessTravel','Department', 
        'EmployeeCount', 'StandardHours', 'NumCompaniesWorked', 
        'JobRole','EducationField','Education','Gender','JobLevel','MaritalStatus','Over18']
for x in col1:
    print('{}: \n{}'.format(x.upper(),df[x].value_counts()), "\n\n")

# Vizualization of the data
## Attrition Pie plot
explode=(0.08,0)

df['Attrition'].value_counts().plot.pie(autopct='%1.2f%%',figsize=(3,3),explode=explode,colors=['#99ff99','#ff6666'])
plt.title("Pie plot of attrition", fontsize=14)
plt.tight_layout()
plt.legend()
plt.show()

labels = df['Attrition'].unique()
sizes = df['Attrition'].value_counts()
colors=['#99ff99','#ff6666']
explode = (0.05,0.05)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', pctdistance=0.85, startangle=90, explode=explode)

centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()

## Ratio of Attrition based on Gender
df1 = df.groupby(['Attrition','Gender']).agg({'Gender':'count'})
df1 = df1.rename({'Gender': 'Gender_count'}, axis=1)
df1.reset_index(inplace=True)

print(df1.head())

labels1 = df.Attrition.unique()
size1 = df.Attrition.value_counts()

labels2 = df1.Gender
size2 = df1.Gender_count

colors1 = ['#99ff99','#ff6666']
colors2 = ['#ffb3e6','#a6d2ff','#ffb3e6','#a6d2ff']

plt.pie(size1, labels=labels1, colors=colors1, startangle=90,frame=True)
plt.pie(size2, labels=labels2, colors=colors2,radius=0.75,startangle=90)
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
 
plt.axis('equal')
plt.tight_layout()
plt.legend()
plt.show()

plt.figure(figsize=(10,10))
prd_gender=pd.crosstab(df['Attrition'],df['Gender'])

ax=prd_gender.plot(kind='bar')
plt.xticks(rotation=0)
plt.title("Gender and Attrition")
plt.show()

## Ratio of Attrition based on Job satisfaction Level
df1 = df.groupby(['Attrition','JobSatisfaction']).agg({'JobSatisfaction':'count'})
df1 = df1.rename({'JobSatisfaction': 'JobSatisfaction_count'}, axis=1)
df1.reset_index(inplace=True)
print(df1)

labels1 = df.Attrition.unique()
size1 = df.Attrition.value_counts()

labels2 = df1.JobSatisfaction
size2 = df1.JobSatisfaction_count

colors1 = ['#99ff99','#ff6666']
colors2 = ['#bbc0f0','#8c94de','#6c78e6','#051ae3']

plt.pie(size1, labels=labels1, colors=colors1, startangle=90,frame=True)
plt.pie(size2, labels=labels2, colors=colors2,radius=0.75,startangle=90)
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
 
plt.axis('equal')
plt.tight_layout()
plt.title("Job_Satisfaction vs Attrition")
plt.show()

plt.figure(figsize=(10,10))
prd_gender=pd.crosstab(df['Attrition'],df['JobSatisfaction'])

ax=prd_gender.plot(kind='bar')
plt.xticks(rotation=0)
plt.title("Job_Satisfaction vs Attrition")

## Ratio of Attrition based on Wrok Life Balance
df1 = df.groupby(['Attrition','WorkLifeBalance']).agg({'WorkLifeBalance':'count'})
df1 = df1.rename({'WorkLifeBalance': 'WorkLifeBalance_count'}, axis=1)
df1.reset_index(inplace=True)
print(df1)

labels1 = df.Attrition.unique()
size1 = df.Attrition.value_counts()

labels2 = df1.WorkLifeBalance
size2 = df1.WorkLifeBalance_count

colors1 = ['#99ff99','#ff6666']
colors2 = ['#bbc0f0','#8c94de','#6c78e6','#051ae3']

plt.pie(size1, labels=labels1, colors=colors1, startangle=90,frame=True)
plt.pie(size2, labels=labels2, colors=colors2,radius=0.75,startangle=90)
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
 
plt.axis('equal')
plt.tight_layout()
plt.title("Work_life_balance vs Attrition")
plt.show()

plt.figure(figsize=(10,10))
prd_gender=pd.crosstab(df['Attrition'],df['WorkLifeBalance'])

ax=prd_gender.plot(kind='bar')
plt.xticks(rotation=0)
plt.title("Work_life_balance vs Attrition")

## Age group showing higher Attrition Rate
df1 = df.groupby(['Attrition','Age']).agg({'Age':'count'})
df1 = df1.rename({'Age': 'Age_count'}, axis=1)
df1.reset_index(inplace=True)
print(df1)

plt.figure(figsize=(15, 5))
x_age = sorted(df['Age'].value_counts().index)
sns.countplot(data=df, x='Age', order=x_age)

plt.title('Age vs age count', fontsize = 30)
plt.xlabel('Ages', fontsize = 15)
plt.ylabel('Count', fontsize = 15)

# Plotting Attrition vs Age barplot and lineplot
fig_dims=(12,4)
fig, ax = plt.subplots(figsize=fig_dims)
sns.countplot(x='Age', hue='Attrition', ax=ax, data=df, edgecolor=sns.color_palette("dark", n_colors=1))

df1 = df.groupby(['Attrition','Age']).agg({'Age':'count'})
df1 = df1.rename({'Age': 'Age_count'}, axis=1)
df1.reset_index(inplace=True)
print(df1)

df1.Attrition.value_counts()

fig, ax =plt.subplots(2,1,figsize=(20,18))

ax[0].bar(df1['Age'].iloc[:43].unique(), df1["Age_count"].iloc[:43])
ax[0].bar(df1['Age'].iloc[43:].unique(), df1["Age_count"].iloc[43:])
ax[0].set_title("Age vs Attrition rate", fontsize=46)
ax[0].set_xlabel('Age', fontsize=25)
ax[0].set_ylabel('Age_count', fontsize=25)
plt.grid(True)

ax[1].plot(df1.Age.iloc[:43], df1.Age_count.iloc[:43], label='Attrition=NO', marker='*')
ax[1].plot(df1.Age.iloc[43:], df1.Age_count.iloc[43:], label='Attrition=YES', marker='*')
ax[1].set_xlabel('Age', fontsize=25)
ax[1].set_ylabel('Age_count', fontsize=25)
plt.legend()

# Data Preparation & Transformation ----------------------------------------------------------------------------------------------------------------------------

# Lable Encoding

df.head()

col = ['Attrition', 'BusinessTravel', 'Department', 
       'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

df.head()

df.info()

# Feature Selection
## Cheking constant feature
var_thres = VarianceThreshold(threshold=0)
var_thres.fit(df)

var_thres.get_support()

print(df.columns[var_thres.get_support()])


constant_columns = [column for column in df.columns
                    if column not in df.columns[var_thres.get_support()]]
print(constant_columns)
print(len(constant_columns))
print("Shape: ", df.shape)

## Correlation among numerical features
pp = sns.pairplot(df[["Age", "MonthlyIncome", "PercentSalaryHike", 
                      "TotalWorkingYears", "YearsAtCompany", 
                      "YearsSinceLastPromotion", "YearsWithCurrManager", 
                      "Attrition"]], hue = "Attrition", 
                  plot_kws=dict(edgecolor="k", linewidth=0.5), diag_kind="kde", diag_kws=dict(shade=True))
t = fig.suptitle('Analyzing Attrition rate of a Company', fontsize=30)

sns.set(font_scale=0.45)
plt.title('Analyzing Attrition rate of a Company')
cmap = sns.diverging_palette(260, 10, as_cmap=True)
sns.heatmap(df[["MonthlyIncome", "NumCompaniesWorked", 
                "PercentSalaryHike", "TotalWorkingYears", 
                "YearsAtCompany", "YearsSinceLastPromotion", 
                "YearsWithCurrManager"]].corr("spearman"), 
            vmax=1.2, annot=True, square='square', cmap=cmap, fmt = '.0%', linewidths=2)

# With the following function we can select highly correlated features
# It will remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr("spearman")
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(df, 0.85)
corr_features

## Relation among categorical features
f_p_values = chi2(df[cat],df["Attrition"])

p_values = pd.Series(f_p_values[1])
p_values.index = cat
p_values.sort_values(ascending=False)

# Null Hypothesis: The null hypothesis states that there is no relationship between the two variables
cnt = 0
for i in p_values:
    if i > 0.05:
        print("There is no relationship", p_values.index[cnt], i)
    else:
        print("There is relationship", p_values.index[cnt], i)
    
    cnt += 1

## One more way to check relation among categorical features i.e. Spearman Correlation

sns.set(font_scale=0.45)
plt.title('Analyzing Attrition rate of a Company')
cmap = sns.diverging_palette(260, 10, as_cmap=True)
sns.heatmap(df[["PerformanceRating", "JobInvolvement", "WorkLifeBalance", 
                "JobSatisfaction", "EnvironmentSatisfaction", "StockOptionLevel", 
                "JobLevel"]].corr("spearman"), 
            vmax=1.2, annot=True, square='square', cmap=cmap, fmt = '.0%', linewidths=2)


## Relation among numerical and classification column
df_anova = df[["EmployeeID", "Age", "MonthlyIncome", "PercentSalaryHike", 
               "TotalWorkingYears", "YearsAtCompany", "YearsSinceLastPromotion", "YearsWithCurrManager", "Attrition"]]
grps = pd.unique(df_anova.Attrition.values)
grps

for i in range(len(df_anova.columns)-1):
    
    d_data = {grp:df_anova[df_anova.columns[i]][df_anova.Attrition == grp] for grp in grps}

    F, p = stats.f_oneway(d_data[0], d_data[1])
    print("P_Value of {} and Attrition".format(df_anova.columns[i]), p)

    if p < 0.05:
        print("There is relation between {} and Attrition \n".format(df_anova.columns[i]))
    else:
        print("There is no relation between {} and Attrition \n".format(df_anova.columns[i]))

sns.scatterplot(x="EmployeeID", y="MonthlyIncome", hue="Attrition", data=df)

df.drop(['Over18', 'EmployeeCount', 'EmployeeID', 'StandardHours', "BusinessTravel", 
         "Department", "Gender", "JobLevel", "StockOptionLevel", "JobInvolvement", "PerformanceRating", "MonthlyIncome"], 
        axis=1, inplace=True)
print(df.columns)
print(len(df.columns))

# Target Feature Separation
x = df.drop("Attrition", axis=1)
y = df.Attrition

print(len(y[y==0]), len(y[y==1]))



# Data Projection ----------------------------------------------------------------------------------------------------------------------------
# Handling Imbalanced data
os =  RandomOverSampler(sampling_strategy=1)

x_res, y_res = os.fit_resample(x, y) 

print(len(y_res[y_res==0]), len(y_res[y_res==1]))
print(len(x_res))

# Scaling the data
scaler = StandardScaler()
features = scaler.fit_transform(x_res)
features

# Logical Test Design------------------------------------------------------------------------------------------------------------------------

# Splitting the data into train and test data
x_train, x_test, y_train, y_test = train_test_split(features, y_res, test_size=0.3, random_state=1) 
x_train
y_train
x_test
y_test

# XGBoost Modeling -------------------------------------------------------------------------------------------------------------------------------------
# XGBoost
## GridSearchCV - Hyperparameter tuning
def xgb_grid_search(X, y):
    # Create a dictionary of all values we want to test
    param_grid = {
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    xgb = XGBClassifier()
    
    #use gridsearch to test all values
    xgb_gscv =  RandomizedSearchCV(estimator = xgb,
                           param_distributions = param_grid,
                           scoring = 'accuracy',
                           cv = cv,
                           n_jobs = -1)
    #fit model to data
    xgb_gscv.fit(X, y)
    
    return xgb_gscv.best_params_

xgb_grid_search(x_train, y_train)

xgb = XGBClassifier(min_child_weight=3, max_depth=10, learning_rate=0.15, gamma=0.4, 
                    colsample_bytree=0.5)
xgb.fit(x_train,y_train)

y_pred_xgb = xgb.predict(x_test)

print(classification_report(y_test, y_pred_xgb))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_xgb))
print("Precision:",metrics.precision_score(y_test, y_pred_xgb))
print("Recall:",metrics.recall_score(y_test, y_pred_xgb))

print(xgb.score(x_train,y_train))
print(xgb.score(x_test,y_test))

xgb_tacc = xgb.score(x_test,y_test)

# Confusion Matrix of XGBoost

cm = metrics.confusion_matrix(y_test, y_pred_xgb, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')

# AUC of XGBoost
y_pred_proba = xgb.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

xgb_auc = auc

# Features importance in making predictions by XGBoost Model
pd.Series(xgb.feature_importances_, index=x.columns).nlargest(20)

feat_importances = pd.Series(xgb.feature_importances_, index=x.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances.plot(kind='barh')

x.columns

cat2 = ["JobRole", "NumCompaniesWorked", "PercentSalaryHike", "YearsAtCompany", 
        "YearsSinceLastPromotion", "YearsWithCurrManager", "EnvironmentSatisfaction", 
        "JobSatisfaction", "WorkLifeBalance"]

f_p_values = chi2(df[cat2],df["MaritalStatus"])

p_values = pd.Series(f_p_values[1])
p_values.index = cat2
p_values.sort_values(ascending=False)

# Null Hypothesis: The null hypothesis states that there is no relationship between the two variables
cnt = 0
for i in p_values:
    if i > 0.05:
        print("There is no relationship", p_values.index[cnt], i)
    else:
        print("There is relationship", p_values.index[cnt], i)
    
    cnt += 1
    
# MaritalStatus vs Attrition
plt.figure(figsize=(10,10))
prd_maritalstatus=pd.crosstab(df['Attrition'],df['MaritalStatus'])

ax=prd_maritalstatus.plot(kind='bar')
plt.xticks(rotation=0)
plt.title("MaritalStatus vs Attrition")

# MaritalStatus vs NumCompaniesWorked
plt.figure(figsize=(10,10))
prd_numcompaniesworked=pd.crosstab(df['MaritalStatus'],df['NumCompaniesWorked'])

ax=prd_numcompaniesworked.plot(kind='bar')
plt.xticks(rotation=0)
plt.title("MaritalStatus vs NumCompaniesWorked")

# MaritalStatus vs YearsAtCompany
plt.figure(figsize=(10,10))
prd_YearsAtCompany=pd.crosstab(df['MaritalStatus'],df['YearsAtCompany'])

ax=prd_YearsAtCompany.plot(kind='bar')
plt.xticks(rotation=0)
plt.title("MaritalStatus vs YearsAtCompany")

# MaritalStatus vs YearsSinceLastPromotion
plt.figure(figsize=(10,10))
prd_YearsSinceLastPromotion=pd.crosstab(df['MaritalStatus'],df['YearsSinceLastPromotion'])

ax=prd_YearsSinceLastPromotion.plot(kind='bar')
plt.xticks(rotation=0)
plt.title("MaritalStatus vs YearsSinceLastPromotion")

# MaritalStatus vs YearsWithCurrManager
plt.figure(figsize=(10,10))
prd_YearsWithCurrManager=pd.crosstab(df['MaritalStatus'],df['YearsWithCurrManager'])

ax=prd_YearsWithCurrManager.plot(kind='bar')
plt.xticks(rotation=0)
plt.title("MaritalStatus vs YearsWithCurrManager")



# SVM modeling ---------------------------------------------------------------------------------------
# SVM
# GridSearchCV - Hyperparameter tuning
def svm_grid_search(X, y):
    #create a dictionary of all values we want to test
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001, 0.4, 0.2, 0.8],
                  'kernel': ['rbf', 'poly', 'sigmoid']}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    svm = SVC()
    
    #use gridsearch to test all values
    svm_gscv = GridSearchCV(estimator = svm,
                           param_grid = param_grid,
                           scoring = 'accuracy',
                           cv = cv,
                           n_jobs = -1)
    #fit model to data
    svm_gscv.fit(X, y)
    
    return svm_gscv.best_params_

svm_grid_search(x_train, y_train)

svm = SVC(gamma=1, C=1, kernel='rbf', probability=True)

svm.fit(x_train, y_train)

y_pred_svm = svm.predict(x_test)

print(svm.score(x_train, y_train))
print(svm.score(x_test, y_test))

# Confusion Matrix of SVM
print(metrics.classification_report(y_test, y_pred_svm))

svm_tacc = svm.score(x_test, y_test)

cm = metrics.confusion_matrix(y_test, y_pred_svm, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')

# AUC od SVM
y_pred_proba = svm.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

svm_auc = auc

# Random Forest Modeling---------------------------------------------------------------------------------
# Random Forest
# GridSearchCV - Hyperparameter

def rf_grid_search(X, y):
    #create a dictionary of all values we want to test
    param_grid = { 
    'n_estimators': [5,10,20,40,50,60,70,80,100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
    }
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Random Forest model
    rf = RandomForestClassifier()
    
    #use gridsearch to test all values
    rf_gscv = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
    #fit model to data
    rf_gscv.fit(X, y)
    
    return rf_gscv.best_params_

rf_grid_search(x_train, y_train)

rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=8, max_features='sqrt')
rf.fit(x_train,y_train)
y_pred2 = rf.predict(x_test)

print(classification_report(y_test, y_pred2))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred2))
print("Precision:",metrics.precision_score(y_test, y_pred2))
print("Recall:",metrics.recall_score(y_test, y_pred2))

print(rf.score(x_train,y_train))
print(rf.score(x_test,y_test))

rf_tacc = rf.score(x_test,y_test)

# Confusion Matrix of Random Forest
cm = metrics.confusion_matrix(y_test, y_pred2, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')

# AUC of Random Forest
y_pred_proba = rf.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

rf_auc = auc
























 





































