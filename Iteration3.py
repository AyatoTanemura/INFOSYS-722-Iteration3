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

# Data Import ------------------------------------------------------------------------------------------------------------------------------------

# Data load
df = pd.read_csv("Data/general_data.csv")

# Data check
df.head()
df.tail()

print("Shape of the dataset:",df.shape,'\n')
print("Total number of rows (excluding column name):",df.shape[0],'\n')
print("Total number of column:",df.shape[1],'\n')
print("Columns name: ", df.columns)

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

col = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
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

# Handling Imbalanced data
os =  RandomOverSampler(sampling_strategy=1)

x_res, y_res = os.fit_resample(x, y) 

print(len(y_res[y_res==0]), len(y_res[y_res==1]))
print(len(x_res))

# Scaling the data
scaler = StandardScaler()
features = scaler.fit_transform(x_res)
features

# Logical Test Design

# Splitting the data into train and test data
x_train, x_test, y_train, y_test = train_test_split(features, y_res, test_size=0.3, random_state=1) 
x_train
y_train
x_test
y_test

































 





































