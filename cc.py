# %% [markdown]
# # Credit Card Fraud Detection
# 
# __Goal:__ 
# 
# Predict the probability of an online credit card transaction being fraudulent, based on different properties of the transactions. 
# 

# %% [markdown]
# ## 1. Setup Environment
# 
# The goal of this section is to:
# - Import all the packages
# - Set the options for data visualizations

# %%
# Data Manipulation
import numpy as np 
import pandas as pd 

# Data Visualization
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Time
import time
import datetime

# Machine Learning
from   sklearn.preprocessing import LabelEncoder, minmax_scale
from   sklearn.ensemble import RandomForestClassifier
from   sklearn.decomposition import PCA
from   sklearn.model_selection import train_test_split, GridSearchCV
from   sklearn.metrics import confusion_matrix , classification_report, accuracy_score, roc_auc_score, plot_roc_curve, precision_recall_curve, plot_precision_recall_curve
from   sklearn.calibration import calibration_curve
from   sklearn.calibration import CalibratedClassifierCV

from   xgboost import XGBClassifier
from   lightgbm import LGBMClassifier

from   imblearn.over_sampling import RandomOverSampler
from   scipy.stats import chi2_contingency,  f_oneway

import gc
import warnings
from   tqdm import tqdm


# Set Options
pd.set_option('display.max_rows', 800)
pd.set_option('display.max_columns', 500)
%matplotlib inline
warnings.filterwarnings("ignore")

# %% [markdown]
# ## 2. Data Overview
# 
# Purpose is to:
# 
# 1. Load the datasets 
# 2. Explore the features
# 
# The data is broken into two files **identity** and **transaction**, which are joined by “TransactionID”. 
# 
# **Note:** Not all transactions have corresponding identity information.
# 
# Load the transaction and identity datasets using pd.read_csv()

# %%
%%time
# Load Data
df_id   = pd.read_csv('Data/train_identity.csv')
df_tran = pd.read_csv('Data/train_transaction.csv')

# %%
# Identitiy Data
df_id.sample(6)

# %% [markdown]
# ### Identity Data Description
# 
# Variables in this table are identity information – network connection information (IP, ISP, Proxy, etc) and digital signature (UA/browser/os/version, etc) associated with transactions.
# They're collected by Vesta’s fraud protection system and digital security partners.
# (The field names are masked and pairwise dictionary will not be provided for privacy protection and contract agreement)
# 
# Categorical Features:
# - DeviceType
# - DeviceInfo
# - id_12 - id_38

# %%
# Transaction Data
df_tran.head()

# %% [markdown]
# ### Transaction Data Description
# - __TransactionDT__: timedelta from a given reference datetime (not an actual timestamp)
# - __TransactionAMT__: transaction payment amount in USD
# - __ProductCD__: product code, the product for each transaction
# - __card1 - card6__: payment card information, such as card type, card category, issue bank, country, etc.
# - __addr__: address
# - __dist__: distance
# - **P_ and (R__) emaildomain**: purchaser and recipient email domain
# - __C1-C14__: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
# - __D1-D15__: timedelta, such as days between previous transaction, etc.
# - __M1-M9__: match, such as names on card and address, etc.
# - __Vxxx__: Vesta engineered rich features, including ranking, counting, and other entity relations.

# %% [markdown]
# ## 3. Optimize Memory Used by Data

# %% [markdown]
# #### Memory occupied by the dataframe (in mb)

# %%
df_id.memory_usage(deep=True).sum() / 1024**2  

# %%
df_tran.memory_usage(deep=True).sum() / 1024**2

# %%
df_tran.dtypes

# %% [markdown]
# Certain features occupy more memory than what is needed to store them. Reducing the memory usage by changing data type will speed up the computations.
# 
# Let's create a function for that:
# 
# - int8 / uint8 : consumes 1 byte of memory, range between -128/127 or 0/255
# - bool : consumes 1 byte, true or false
# - float16 / int16 / uint16: consumes 2 bytes of memory, range between -32768 and 32767 or 0/65535
# - float32 / int32 / uint32 : consumes 4 bytes of memory, range between -2147483648 and 2147483647
# - float64 / int64 / uint64: consumes 8 bytes of memory

# %%
print('int64 min: ', np.iinfo(np.int64).min)
print('int64 max: ', np.iinfo(np.int64).max)

# %%
print('int8 min: ', np.iinfo(np.int8).min)
print('int8 max: ', np.iinfo(np.int8).max)

# %%
# Reduce memory usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# %% [markdown]
# Use the defined function to reduce the memory usage

# %%
# Reduce the memory size of the dataframe
df_id   = reduce_mem_usage(df_id)
df_tran = reduce_mem_usage(df_tran)

# %% [markdown]
# ## 4. Basic Data Stats
# 
# Before attempting to solve the problem, it's very important to have a good understanding of data.
# 
# The goal of this section is to:
# - Get the dimensions of data
# - Get the summary of data
# - Get various statistics of data

# %% [markdown]
# #### Shape of dataframe

# %%
# Dimensions of identity dataset
print(df_id.shape)

# %% [markdown]
# The dataset has 144233 rows and 41 columns

# %%
# Dimensions of transaction dataset
print(df_tran.shape)

# %% [markdown]
# The dataset has 590540 rows and 394 columns

# %% [markdown]
# __Check how many transactions has ID info__

# %%
# How many had ID info?
df_tran.TransactionID.isin(df_id.TransactionID).sum()

# %% [markdown]
# #### Summary of dataframe

# %%
df_id.head()

# %%
from pandas_summary import DataFrameSummary
df_id_summary = DataFrameSummary(df_id)
df_id_summary.summary()

# %% [markdown]
# By looking at the summary of datasets, it's clear there is a lot of missing values in the dataset. 
# 
# Let's get missing value stats and various other stats of columns in dataframe. 

# %% [markdown]
# #### Stats on Transaction Dataset
# 

# %%
from pandas_summary import DataFrameSummary
df_tran_summary = DataFrameSummary(df_tran)
df_tran_summary.summary()

# %% [markdown]
# __Check class imbalance__

# %%
df_tran.loc[:, 'isFraud'].value_counts()

# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'> Inferences:</h3>
# </div>
# 
# 
# Lot of interesting things can be observed here:
# 
# - Rows in identity dataset are less than transaction dataset, that means only a subset of transactions in transactions dataset has identity data 
# - Both datasets have the common and unique key as TransactionID, both can be joined at this unique key
# - id_24, id_25, dist2, D7 and many more columns have 90%+ missing values, which means that these columns are probably useless so need to drop it for now
# - Columns from V1 to V339 in transaction dataset are numeric whereas columns from id_01 to id_39 are of mixed datatype
# - TransactionDT column is a timedelta from a given reference datetime (not an actual timestamp). But reference datetime is not known, so need to assume it and convert it to date format
# - Target class is imbalanced. So no need to drop the columns where one category contains the majority of rows 

# %% [markdown]
# ## 5. Data Preprocessing for EDA
# 
# The goal of this section is to:
# - Merge two datasets
# - Drop the columns based on the inferences from previous section
# - Create date features from transaction date 
# 
# Let's start with the first task to merge datasets to form one. 
# 
# #### Merge the datasets

# %%
# Merge transaction dataset and identity dataset 
df = df_tran.merge(df_id, how='left', left_index=True, right_index=True, on='TransactionID')

del df_tran, df_id

gc.collect()

# %% [markdown]
# Get dimensions of training dataset

# %%
# Dimentions of data
df.shape

# %% [markdown]
# Since left join was performed on transaction dataset, number of rows are same as transaction dataset.

# %% [markdown]
# #### Add missing flag

# %%
# Add flag column for missing values
for col in df.columns:
    df[col+"_missing_flag"] = df[col].isnull()
    
df.head()

# %% [markdown]
# #### Clean Data
# 
# Let's drop the columns which may not be useful for our analysis
# 
# Create a missing value flag column for the columns we are dropping which have more than 90% missing values, there might be some specific pattern associated with missing values and transaction being fraud

# %%
# Drop the columns where one category contains more than 90% values
drop_cols = []

for col in df.columns:
    missing_share = df[col].isnull().sum()/df.shape[0]
    if missing_share > 0.9:
        drop_cols.append(col)
        print(col)
        # df[col + "_missing_flag"] = df[col].isnull()
    
good_cols = [col for col in df.columns if col not in drop_cols]    

# %% [markdown]
# Remove the columns which doesn't having any variance

# %%
# Drop the columns which have only one unique value
drop_cols = []
for col in good_cols:
    unique_value = df[col].nunique()
    if unique_value == 1:
        drop_cols.append(col)
        print(col)
good_cols = [col for col in good_cols if col not in drop_cols]

# %% [markdown]
# Filter the dataset with only good columns 

# %%
# Filter the data for relevant columns only
df = df[good_cols]

# %% [markdown]
# Get dimentions of training dataset

# %%
# Dimentions of data
df.shape

# %% [markdown]
# #### Create date features
# 
# Let's create date features from TransactionDT features

# %%
# Date features
START_DATE         = '2017-12-01'
startdate          = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
df["Date"]         = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

df['_Weekdays']    = df['Date'].dt.dayofweek
df['_Hours']       = df['Date'].dt.hour
df['_Days']        = df['Date'].dt.day

# %%
df = reduce_mem_usage(df)

# %% [markdown]
# ## 6. Exploratory Data Analysis
# 
# Exploratory data analysis is an approach to analyze or investigate data sets to find out patterns and see if any of the variables can be useful to explain / predict the y variables. 
# 
# Visual methods are often used to summarise the data. Primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing tasks.
# 
# The goal of this section is to:
# - Check if the target variable is balanced or is there a need to balance the target variable
# - Analyze the transaction amount
# - Get insights or relationships from the data which would be useful from business perspective.
# 
# ### Check distribution of target variable

# %%
# Get count of target class
df['isFraud'].value_counts()

# %% [markdown]
# Let's check the distribution of target class using a bar plot and check the proportion of transactions amounts being fraud

# %%
# Draw a countplot to check the distribution of target variable
df['TransactionAmt'] = df['TransactionAmt'].astype(float)
total = len(df)
total_amt = df.groupby(['isFraud'])['TransactionAmt'].sum().sum()
plt.figure(figsize=(16,6))

plt.subplot(121)
g = sns.countplot(x='isFraud', data=df )
g.set_title("Fraud Transactions Distribution \n 0: No Fraud | 1: Fraud", fontsize=18)
g.set_xlabel("Is fraud?", fontsize=18)
g.set_ylabel('Count', fontsize=18)
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=15) 

perc_amt = (df.groupby(['isFraud'])['TransactionAmt'].sum())
perc_amt = perc_amt.reset_index()

plt.subplot(122)
g1 = sns.barplot(x='isFraud', y='TransactionAmt',  dodge=True, data=perc_amt)
g1.set_title("% Total Amount in Transaction Amt \n 0: No Fraud | 1: Fraud", fontsize=18)
g1.set_xlabel("Is fraud?", fontsize=18)
g1.set_ylabel('Total Transaction Amount Scalar', fontsize=18)
for p in g1.patches:
    height = p.get_height()
    g1.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total_amt * 100),
            ha="center", fontsize=15) 
    
plt.show()

# %%
# Average transaction amount by Y
df.groupby('isFraud')['TransactionAmt'].mean()

# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'> Inferences:</h3>
# </div>

# %% [markdown]
# - The target variable is **imbalanced**. 3.5% transactions are Fraud
# - Around same % of transaction amounts are fraud
# 
# 
# Let's explore the Transaction amount further
# 
# ### Check distribution of Transaction Amount

# %%
# Distribution plot of Transaction Amount
plt.figure(figsize=(16,12))

sns.distplot(df['TransactionAmt'])
plt.title("Transaction Amount Distribution",fontsize=18)
plt.ylabel("Probability")

# %% [markdown]
# There are certain transactions which are of very high amount, let's remove those to check the distribution

# %%
# Distribution plot of Transaction Amount less than 1000
plt.figure(figsize=(16,12))

plt.suptitle('Transaction Values Distribution', fontsize=22)
sns.distplot(df[df['TransactionAmt'] <= 1000]['TransactionAmt'])
plt.title("Transaction Amount Distribuition <= 1000", fontsize=18)
plt.xlabel("Transaction Amount", fontsize=15)
plt.ylabel("Probability", fontsize=15)

plt.show()

# %% [markdown]
# Most transactions lie in < $200 range

# %% [markdown]
# Transaction amount is right skewed. 
# 
# Let's look at the log of transaction amount

# %%
# Distribution plot of Transaction Amount less than 1000
plt.figure(figsize=(16,12))

plt.suptitle('Transaction Values Distribution', fontsize=22)
sns.distplot(np.log(df['TransactionAmt']))
plt.title("Transaction Amount (Log) Distribuition", fontsize=18)
plt.xlabel("Transaction Amount", fontsize=15)
plt.ylabel("Probability", fontsize=15)

plt.show()

# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'> Inferences:</h3>
# </div>
# 
# - Transaction Amount is right skewed.
# - Log of transaction amount is almost normally distributed, so use log of transaction amount while building the model
# 
# ### Product Features
# 
# - Distribution of ProductCD
# - Distribution of Frauds by Product

# %%
def plot_cat_feat_dist(df, col):
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    plt.figure(figsize=(16,12))
    plt.suptitle(f'{col} Distributions', fontsize=22)

    plt.subplot(221)
    g = sns.countplot(x=col, data=df, order=tmp[col].values)

    g.set_title(f"{col} Distribution", fontsize=16)
    g.set_xlabel(f"{col} Name", fontsize=17)
    g.set_ylabel("Count", fontsize=17)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=14) 

    plt.subplot(222)
    g1 = sns.countplot(x=col, hue='isFraud', data=df, order=tmp[col].values)
    plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])
    gt = g1.twinx()
    gt = sns.pointplot(x=col, y='Fraud', data=tmp, color='black', order=tmp[col].values, legend=False)
    gt.set_ylabel("% of Fraud Transactions", fontsize=16)

    g1.set_title(f"{col} Distribution by Target Variable (isFraud) ", fontsize=16)
    g1.set_xlabel(f"{col} Name", fontsize=17)
    g1.set_ylabel("Count", fontsize=17)


    plt.subplots_adjust(hspace = 0.4, top = 0.85)

    plt.show()

# %%
plot_cat_feat_dist(df, "ProductCD")

# %%
# Average fraud per transaction by ProductCD
df.groupby('ProductCD')['isFraud'].mean()

# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'> Inferences:</h3>
# </div>
# 
# - 75% of the transactions are for Product Catergory W
# - 11.6% of the transactions are for Product Category C
# - Fraud Transaction rate is maximum for Product Category C and minimum for Product Category W

# %% [markdown]
# ### Card Features

# %%
# Card 4
plot_cat_feat_dist(df, "card4")

# %%
# Average fraud per transaction by Card4
df.groupby('card4')['isFraud'].mean()

# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'> Inferences:</h3>
# </div>
# 
# - 97% of transactions are from Mastercard(32%) and Visa(65%
# - Fraud transaction rate is highest for discover cards(~8%) against ~3.5% of Mastercard and Visa and 2.87% in American Express

# %%
# Card 6
plot_cat_feat_dist(df, "card6")

# %%
# Average fraud per transaction by Card6
df.groupby('card6')['isFraud'].mean()

# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'> Inferences:</h3>
# </div>
# 
# - Almost all the transactions are from Credit and Debit cards. 
# - Debit card transactions are almost 3 times as compared to credit card transactions.
# - Fraud transaction rate is high for Credit cards as compared to Debit cards.
# 

# %% [markdown]
# ### P emaildomain
# 
# - It has multiple domains, let's group them by the respective enterprises
# - Set all values with less than 500 entries as "Others"

# %%
df.loc[df['P_emaildomain'].isin(['gmail.com', 'gmail']),'P_emaildomain'] = 'Google'

df.loc[df['P_emaildomain'].isin(['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk',
                                         'yahoo.co.jp', 'yahoo.de', 'yahoo.fr',
                                         'yahoo.es']), 'P_emaildomain'] = 'Yahoo Mail'
df.loc[df['P_emaildomain'].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 
                                         'hotmail.es','hotmail.co.uk', 'hotmail.de',
                                         'outlook.es', 'live.com', 'live.fr',
                                         'hotmail.fr']), 'P_emaildomain'] = 'Microsoft'
df.loc[df.P_emaildomain.isin(df.P_emaildomain\
                                         .value_counts()[df.P_emaildomain.value_counts() <= 500 ]\
                                         .index), 'P_emaildomain'] = "Others"
df.P_emaildomain.fillna("NoInf", inplace=True)

# %%
def plot_cat_with_amt(df, col, lim=2000):
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    
    plt.figure(figsize=(16,14))    
    plt.suptitle(f'{col} Distributions ', fontsize=24)
    
    plt.subplot(211)
    g = sns.countplot( x=col,  data=df, order=list(tmp[col].values))
    gt = g.twinx()
    gt = sns.pointplot(x=col, y='Fraud', data=tmp, order=list(tmp[col].values),
                       color='black', legend=False, )
    gt.set_ylim(0,tmp['Fraud'].max()*1.1)
    gt.set_ylabel("%Fraud Transactions", fontsize=16)
    g.set_title(f"Share of {col} categories and % of Fraud Transactions", fontsize=18)
    g.set_xlabel(f"{col} Category Names", fontsize=16)
    g.set_ylabel("Count", fontsize=17)
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    sizes = []
    for p in g.patches:
        height = p.get_height()
        sizes.append(height)
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center",fontsize=12) 
        
    g.set_ylim(0,max(sizes)*1.15)
    
    #########################################################################
    perc_amt = (df.groupby(['isFraud',col])['TransactionAmt'].sum() \
                / df.groupby([col])['TransactionAmt'].sum() * 100).unstack('isFraud')
    perc_amt = perc_amt.reset_index()
    perc_amt.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    amt = df.groupby([col])['TransactionAmt'].sum().reset_index()
    perc_amt = perc_amt.fillna(0)
    plt.subplot(212)
    g1 = sns.barplot(x=col, y='TransactionAmt', 
                       data=amt, 
                       order=list(tmp[col].values))
    g1t = g1.twinx()
    g1t = sns.pointplot(x=col, y='Fraud', data=perc_amt, 
                        order=list(tmp[col].values),
                       color='black', legend=False, )
    g1t.set_ylim(0,perc_amt['Fraud'].max()*1.1)
    g1t.set_ylabel("%Fraud Total Amount", fontsize=16)
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    g1.set_title(f"Transactions amount by {col} categories and % of Fraud Transactions (Amounts)", fontsize=18)
    g1.set_xlabel(f"{col} Category Names", fontsize=16)
    g1.set_ylabel("Transaction Total Amount(U$)", fontsize=16)
    g1.set_xticklabels(g.get_xticklabels(),rotation=45)    
    
    for p in g1.patches:
        height = p.get_height()
        g1.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total_amt*100),
                ha="center",fontsize=12) 
        
    plt.subplots_adjust(hspace=.4, top = 0.9)
    plt.show()
    

# %%
plot_cat_with_amt(df, 'P_emaildomain')

# %%
# Average fraud per transaction by Card6
df.groupby('P_emaildomain')['isFraud'].mean()

# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'> Inferences:</h3>
# </div>
# 
# - Majority of transactions are with P_emaildomain as Google, Microsoft and Yahoo Mail
# - There isn't any information about P_emaildomain of around 16% transactions in terms of count and 14.11% in terms of amount
# - Fraud transaction rate for Microsoft is high as compared to Google and Yahoo mail 
# - Fraud transaction rate (amount) for Google is high as comapred to Microsoft and Yahoo mail

# %% [markdown]
# ### R-Email Domain
# 
# - It has multiple domains, let's group them by the respective enterprises
# - Set all values with less than 500 entries as "Others"

# %%
df.loc[df['R_emaildomain'].isin(['gmail.com', 'gmail']),'R_emaildomain'] = 'Google'

df.loc[df['R_emaildomain'].isin(['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk',
                                             'yahoo.co.jp', 'yahoo.de', 'yahoo.fr',
                                             'yahoo.es']), 'R_emaildomain'] = 'Yahoo Mail'
df.loc[df['R_emaildomain'].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 
                                             'hotmail.es','hotmail.co.uk', 'hotmail.de',
                                             'outlook.es', 'live.com', 'live.fr',
                                             'hotmail.fr']), 'R_emaildomain'] = 'Microsoft'
df.loc[df.R_emaildomain.isin(df.R_emaildomain\
                                         .value_counts()[df.R_emaildomain.value_counts() <= 300 ]\
                                         .index), 'R_emaildomain'] = "Others"
df.R_emaildomain.fillna("NoInf", inplace=True)

# %%
plot_cat_with_amt(df, 'R_emaildomain')

# %%
# Average fraud per transaction by Card6
df.groupby('R_emaildomain')['isFraud'].mean()

# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'> Inferences:</h3>
# </div>
# 
# - There isn't any information about R_emaildomain for Majority of transactions (76.75% count , 85.62% amount)
# - Fraud transaction rate for Google is high as compared to Yahoo, anaonymous.com and Microsoft

# %% [markdown]
# ###  Days of the Month
# 
# Reference date is not known, it has been assumed. So can't say concretely if the day number is correct

# %%
plot_cat_with_amt(df, '_Days')

# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'> Inferences:</h3>
# </div>
# 
# - The perc of fraud transactions is highest towards the beginning and the end of the month. Might be accelerated at the time of receiving pay-checks.
# 
# - Incidentally, fraud transaction rate is high on the days when number of transactions are less
# 
# - Day 29,30 and 31 are having less transactions, looks like people are cautious with spending in those times.

# %% [markdown]
# ### Days of the week
# 
# Reference date is not known, it has been assumed. So can't say concretely if the day number is correct

# %%
plot_cat_with_amt(df, '_Weekdays')

# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'> Inferences:</h3>
# </div>
# 
# - Surprisingly fraud transaction rate is high on the days when number of transactions and transaction amounts are less. Day 0 and 6
# - Day 0 and 6 have less transactions, these might be weekend days

# %% [markdown]
# ### Hour of the Day

# %%
plot_cat_with_amt(df, '_Hours')

# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'> Inferences:</h3>
# </div>
# 
# - Transactions start decreasing mid night but the fraud rate starts increasing
# - Transactions from 3 AM to 12 PM needs to monitored very closely 

# %% [markdown]
# ### Device Type

# %%
plot_cat_with_amt(df, "DeviceType")

# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'> Inferences:</h3>
# </div>
# 
# - Device type is known for only 24% of the transactions
# - Due to lack of data points, we can't infer from this analysis

# %% [markdown]
# ### Columns from identity data

# %%
for col in ['id_12', 'id_15', 'id_16', 'id_28', 'id_29']:
    df[col] = df[col].fillna('NaN')
    plot_cat_with_amt(df, col)

# %%
df.loc[df['id_30'].str.contains('Windows', na=False), 'id_30'] = 'Windows'
df.loc[df['id_30'].str.contains('iOS', na=False), 'id_30'] = 'iOS'
df.loc[df['id_30'].str.contains('Mac OS', na=False), 'id_30'] = 'Mac'
df.loc[df['id_30'].str.contains('Android', na=False), 'id_30'] = 'Android'
df['id_30'].fillna("NAN", inplace=True)

plot_cat_with_amt(df, "id_30")

# %%
df.loc[df['id_31'].str.contains('chrome', na=False), 'id_31'] = 'Chrome'
df.loc[df['id_31'].str.contains('firefox', na=False), 'id_31'] = 'Firefox'
df.loc[df['id_31'].str.contains('safari', na=False), 'id_31'] = 'Safari'
df.loc[df['id_31'].str.contains('edge', na=False), 'id_31'] = 'Edge'
df.loc[df['id_31'].str.contains('ie', na=False), 'id_31'] = 'IE'
df.loc[df['id_31'].str.contains('samsung', na=False), 'id_31'] = 'Samsung'
df.loc[df['id_31'].str.contains('opera', na=False), 'id_31'] = 'Opera'
df['id_31'].fillna("NAN", inplace=True)
df.loc[df.id_31.isin(df.id_31.value_counts()[df.id_31.value_counts() < 200].index), 'id_31'] = "Others"
plot_cat_with_amt(df, "id_31")


# %% [markdown]
# ### Get column names

# %%
cat_columns = df.select_dtypes(include=['object']).columns
len(cat_columns)

# %%
binary_columns = [col for col in df.columns if df[col].nunique() == 2]
len(binary_columns)

# %%
num_columns = [col for col in df.columns if (col not in cat_columns) & (col not in binary_columns)]
len(num_columns)

# %%
cat_columns = cat_columns.to_list() + binary_columns

# %% [markdown]
# ## 7. Statistical Significance test
# 
# ### Chi square test for categorical columns

# %%
from   scipy.stats import chi2_contingency

# %%
# significance value
alpha = 0.05

significant_categorical_variables = []

for col in cat_columns:  
    # Create a crosstab table
    temp = pd.crosstab(df[col],df['isFraud'].astype('category'))
    
    # Get chi-square value , p-value, degrees of freedom, expected frequencies using the function chi2_contingency
    stat, p, dof, expected = chi2_contingency(temp)
    
    # Determine whether to reject or keep your null hypothesis
    print(col.ljust(40), ',  chisquared=%.5f,   p-value=%.5f' % (stat, p))
    if p <= alpha:
        significant_categorical_variables.append(col)
    else:
        ""

# %%
# Significant variables
# print(significant_categorical_variables)

# %% [markdown]
# ### Calculate odds

# %% [markdown]
# Chi-Square test tells if the entire variable is useful or not. 

# %%
ctab = pd.crosstab(df['ProductCD'], df['isFraud'].astype('category'))
ctab

# %% [markdown]
# #### Odds

# %%
ctab.columns = ctab.columns.add_categories('odds')
ctab['odds'] = ctab[1]/ctab[0]
ctab

# %% [markdown]
# #### Odds Ratio

# %%
ctab.columns = ctab.columns.add_categories('odds_ratio')
ctab['odds_ratio'] = ctab['odds'] / (ctab[1].sum()/ctab[0].sum())
ctab

# %% [markdown]
# Highers odds ratio implies more chance of fraud in that category. 
# 
# Farther away it is from 1.0 (both directions) more important the variable is.

# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ## 8. ANOVA Test

# %%
from scipy.stats import f_oneway

# %%
# significance value
alpha = 0.05

significant_numerical_variables = []
for col in num_columns[2:]:
    # Determine whether to reject or keep your null hypothesis
    if df.loc[:, col].nunique() > 50:
        F, p = f_oneway(df[df.isFraud == 1][col].dropna(),
                    df[df.isFraud == 0][col].dropna())
        print(col.ljust(40), ',   F-statistic=%.5f, p=%.5f' % (F, p), df.loc[:, col].nunique())
        if p <= alpha:
            significant_numerical_variables.append(col)

# %%
# Significant variables
# significant_numerical_variables

# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'> EDA Inferences:</h3>
# </div>
# 
# - The target class in imbalanced
# - Only 3.5% transactions are fraud in terms of count and 3.87% in terms of transaction amount
# - `TransactionAmt` is right skewed so log transform needs to be used to make it normally distributed  
# - Fraud Transaction rate is maximum for Product Category C and minimum for Product Category W
# - 97% of transactions are from Mastercard(32%) and Visa(65%
# - Fraud transaction rate is highest for discover cards(~8%) against ~3.5% of Mastercard and Visa and 2.87% in American Express
# - Almost all the transactions are from Credit and Debit cards. 
# - Debit card transactions are almost 3 times as compared to credit card transactions.
# - Fraud transaction rate is high for Credit cards as compared to Debit cards.
# - Fraud transaction rate for Microsoft is high as compared to Google and Yahoo mail  # p emaildomain
# - Fraud transaction rate (amount) for Google is high as comapred to Microsoft and Yahoo mail #p emaildomain
# - There isn't any information about R_emaildomain for Majority of transactions (76.75% count , 85.62% amount) #r emaildomain
# - Fraud transaction rate for Google is high as compared to Yahoo, anaonymous.com and Microsoft #r emaildomain
# - Surprisingly fraud transaction rate is high on the days when number of transactions are less
# - Day 29,30 and 31 are having less transactions, looks like people are broke at the month end 
# - Surprisingly fraud transaction rate is high on the days when number of transactions and transaction amounts are less. Day 0 and 6
# - Day 0 and 6 have less transactions, these might be weekend days
# - Transactions start decreasing mid night but the fraud rate starts increasing
# - Transactions from 3 AM to 12 PM needs to monitored very closely 

# %% [markdown]
# ## Mini Challenge
# 
# - Analyze the columns having majority of missing values. Create a new column which has a flag whether the value is missing or not, later on try to find out some pattern of missing data with the target class 

# %% [markdown]
# ## 7. Feature Engineering
# 
# Feature engineering is the process of using domain and statistical knowledge to extract features from raw data via data mining techniques. 
# 
# These features often help to improve the performance of machine learning models.
# 
# 
# The goal of this section is to:
# - Engineer domain specific features
# - Dimensionality reduction
# - Encode the categorical features

# %%
df.head()

# %% [markdown]
# ### Domain Specific Features
# 
# You need to engineer the domain specific features. This might boost up the predictive power. This often gives better performing models
# 
# Domain knowledge is one of the key pillars of data science. So always understand the domain before attempting the problem.

# %%
# Transaction amount minus mean of transaction 
df['Trans_min_mean'] = df['TransactionAmt'] - np.nanmean(df['TransactionAmt'],dtype="float64")
df['Trans_min_std']  = df['Trans_min_mean'] / np.nanstd(df['TransactionAmt'].astype("float64"),dtype="float64")

# %% [markdown]
# Replace value by the group's mean (or standard dev)

# %%
# Features for transaction amount and card 
df['TransactionAmt_to_mean_card1'] = df['TransactionAmt'] / df.groupby(['card1'])['TransactionAmt'].transform('mean')
df['TransactionAmt_to_mean_card4'] = df['TransactionAmt'] / df.groupby(['card4'])['TransactionAmt'].transform('mean')
df['TransactionAmt_to_std_card1']  = df['TransactionAmt'] / df.groupby(['card1'])['TransactionAmt'].transform('std')
df['TransactionAmt_to_std_card4']  = df['TransactionAmt'] / df.groupby(['card4'])['TransactionAmt'].transform('std')

# %%
# Log of transaction amount
df['TransactionAmt'] = np.log(df['TransactionAmt'])

# %%
df.head()

# %%
# Save train df to csv file 
# df.to_csv("Intermediate_Datasets/df_intermediate1.csv",index = False)

# Read train df
df = pd.read_csv("Intermediate_Datasets/df_intermediate1.csv")

# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ## 8. Dimensionality Reduction - PCA
# 
# When dealing with high dimensional data, it is often useful to reduce the dimensionality by projecting the data to a lower dimensional subspace which captures the “essence” of the data.
# 
# Dimensionality reduction, or dimension reduction, is the transformation of data from a high-dimensional space into a low-dimensional space so that the low-dimensional representation retains some meaningful properties of the original data, ideally close to its intrinsic dimension.
# 
# 
# **Principal component analysis**  is a technique for reducing the dimensionality of such datasets, increasing interpretability but at the same time minimizing information loss. It does so by creating new uncorrelated variables that successively maximize variance.

# %%
# initialize function to perform PCA
def perform_PCA(df, cols, n_components, prefix='PCA_', rand_seed=4):
    pca = PCA(n_components=n_components, random_state=rand_seed)
    principalComponents = pca.fit_transform(df[cols])
    principalDf = pd.DataFrame(principalComponents)
    df.drop(cols, axis=1, inplace=True)

    principalDf.rename(columns=lambda x: str(prefix)+str(x), inplace=True)
    df = pd.concat([df, principalDf], axis=1)
    return df

# %% [markdown]
# Create a list of all the columns on which PCA needs to performed

# %%
# Columns starting from V1 to V339
filter_col = df.columns[53:392]

# %% [markdown]
# Impute missing values in the mas_v columns, later use minmax_scale function to scale the values in these columns

# %%
from   sklearn.preprocessing import minmax_scale

# %%
# Fill na values and scale V columns
for col in filter_col:
    df[col] = df[col].fillna((df[col].min() - 2))
    df[col] = (minmax_scale(df[col], feature_range=(0,1)))

# Perform PCA    
df          = perform_PCA(df, filter_col, prefix='PCA_V_', n_components=30)

# %% [markdown]
# Reduce memory usage of df as lot of new features have been created

# %%
df = reduce_mem_usage(df)

# %%
df.head()

# %%
# Plot first 2 PCA features and colour by target variable
plt.figure(figsize=(12, 8));
groups = df.groupby("isFraud")
for name, group in groups:
    plt.scatter(group["PCA_V_0"], group["PCA_V_1"], label=name)
plt.legend()
plt.show()

# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ## 9. Feature Encoding
# 
# 
# Encoding is the process of converting data from one form to another. Most of the Machine learning algorithms can not handle categorical values unless we convert them to numerical values. Many algorithm’s performances vary based on how Categorical columns are encoded.
# - **Frequency Encoding**  - It is a way to utilize the frequency of the categories as labels. In the cases where the frequency is related somewhat with the target variable, it helps the model to understand and assign the weight in direct and inverse proportion, depending on nature of the data.

# %% [markdown]
# Create a list of variables that needs to be encoded using frequency encoding. Let's note down the features which has more than 30 unique values,  We would using frequency encoding for these features only 

# %%
cat_columns = df.select_dtypes(include=['object']).columns
len(cat_columns)

# %%
binary_columns = [col for col in df.columns if df[col].nunique() == 2]
len(binary_columns)

# %%
num_columns = [col for col in df.columns if (col not in cat_columns) & (col not in binary_columns)]
len(num_columns)

# %%
cat_columns = cat_columns.to_list() + binary_columns

# %%
# Frequecny encoding variables
frequency_encoded_variables = []
for col in cat_columns:
    if df[col].nunique() > 30:
        print(col, df[col].nunique())
        frequency_encoded_variables.append(col)

# %% [markdown]
# It's time to encode the variables using frequency encoding

# %%
# Frequecny enocde the variables
for variable in tqdm(frequency_encoded_variables):
    # group by frequency 
    fq = df.groupby(variable).size()/len(df)    
    # mapping values to dataframe 
    df.loc[:, "{}".format(variable)] = df[variable].map(fq)   
    cat_columns.remove(variable)

# %%
df.head()

# %% [markdown]
# - **Label encoding** - Label Encoding refers to converting the labels into numeric form so as to convert it into the machine-readable form. Machine learning algorithms can then decide in a better way on how those labels must be operated. It is an important pre-processing step for the structured dataset in supervised learning.
# 
# It is a popular encoding technique for handling categorical variables. In this technique, each label is assigned a unique integer based on alphabetical ordering.
# 

# %%
# Label encode the variables
for col in cat_columns:
    lbl        = LabelEncoder()
    lbl.fit(list(df[col].values))
    df[col] = lbl.transform(list(df[col].values))

# %% [markdown]
# Let's reduce the memory usage as lot of new columns has been added to the data frame

# %%
# Reduce memory usage
df = reduce_mem_usage(df)

# %% [markdown]
# **Tip : Save the train df, and clean all memory**

# %%
# Save train df to csv file 
df.to_csv("Intermediate_Datasets/df_intermediate2.csv", index = False)

# %% [markdown]
# ## 10. Data Preprocessing for Model Building
# 
# 
# The goal of this section is to:
# - Clean up columns
# - Create X and y
# - Split the dataset in training and test sets

# %%
# Read train df
df = pd.read_csv("Intermediate_Datasets/df_intermediate2.csv")
# df = df.sample(10000, random_state=0)

# %%
df.loc[:, 'isFraud'].value_counts()

# %% [markdown]
# Drop the columns which may not be useful for model building

# %%
df = df.drop(['TransactionID','TransactionDT','Date'], axis=1)

# %% [markdown]
# Separate the x variables and y variables

# %%
# Split the y variable series and x variables dataset
X = df.drop(['isFraud'],axis=1)
y = df.isFraud.astype(bool)

# Delete train df
del df

# Collect garbage
gc.collect()

# %% [markdown]
# Split the dataset into train set and test set. Train set will be used to train the model. Test set will be used to check the performance of model

# %%
# Split the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# %%
# Head of X_train
X_train.head()

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ## 11. Model Building
# 
# Finally, model building starts here.
# 
# The goal of this section is to:
# - Build ML models
# - Evaluate the performance

# %% [markdown]
# ## Start building the ML models
# 
# Let's start with XGBoost first 

# %% [markdown]
# ## 12. XGBoost Classifier
# 
# XGBoost is an optimized distributed gradient boosting model designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way.

# %%
%%time
# Define the model
xgb = XGBClassifier(nthread = -1, random_state=0)

# Train the model
xgb.fit(X_train, y_train)

xgb

# %% [markdown]
# Let's use the model to get predictions on test dataset. We would be looking at the predicted class and predicted probability both in order to evaluate the performance of the model

# %%
# Prediction
y_pred_xgb = xgb.predict(X_test)
y_prob_pred_xgb = xgb.predict_proba(X_test)
y_prob_pred_xgb = [x[1] for x in y_prob_pred_xgb]
print("Y predicted : ",y_pred_xgb)
print("Y probability predicted : ",y_prob_pred_xgb[:5])

# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ## 13. Evaluation Metrics
# 
# - Accuracy Score
# - Confusion Matrix
# - Classification Report
# - AUC Score
# - Concordance Index
# - ROC curve
# - PR curve 

# %% [markdown]
# Concordance

# %%
from bisect import bisect_left, bisect_right

def concordance(actuals, preds):
    ones_preds  = [p for a,p in zip(actuals, preds) if a == 1]
    zeros_preds = [p for a,p in zip(actuals, preds) if a == 0]
    n_ones      = len([x for x in actuals if x == 1])
    n_total_pairs =  float(n_ones) * float(len(actuals) - n_ones)
    # print("Total Pairs: ", n_total_pairs)

    zeros_sorted = sorted(zeros_preds)

    conc = 0; disc = 0; ties = 0;
    for i, one_pred in enumerate(ones_preds):
        cur_conc = bisect_left(zeros_sorted, one_pred)
        cur_ties = bisect_right(zeros_sorted, one_pred) - cur_conc
        conc += cur_conc
        ties += cur_ties
        disc += float(len(zeros_sorted)) - cur_ties - cur_conc

    concordance = conc/n_total_pairs
    discordance = disc/n_total_pairs
    ties_perc = ties/n_total_pairs
    return concordance

# %% [markdown]
# All evaluation metrics

# %%
def compute_evaluation_metric(model, x_test, y_actual, y_predicted, y_predicted_prob):
    print("\n Accuracy Score : ",accuracy_score(y_actual,y_predicted))
    print("\n AUC Score : ", roc_auc_score(y_actual, y_predicted_prob))
    print("\n Confusion Matrix : \n",confusion_matrix(y_actual, y_predicted))
    print("\n Classification Report : \n",classification_report(y_actual, y_predicted))
    print("\n Concordance Index : ", concordance(y_actual, y_predicted_prob))

    print("\n ROC curve : \n")
    plot_roc_curve(model, x_test, y_actual)
    plt.show() 

    print("\n PR curve : \n")
    plot_precision_recall_curve(model, x_test, y_actual)
    plt.show() 

# %%
concordance(y_test.values, y_prob_pred_xgb)

# %%
# Compute Evaluation Metric
compute_evaluation_metric(xgb, X_test, y_test, y_pred_xgb, y_prob_pred_xgb)

# %% [markdown]
# ## 14. Capture Rates and Calibration Curve
# 
# 
# Divide the data in 10 equal bins as per predicted probability scores. Then, compute the percentage of the total target class 1 captured in every bin. 
# 
# Ideally the proportion should be decreasing as we go down ever bin.
# Let's check it out

# %% [markdown]
# #### Create validation set

# %%
# Create Validation set
validation_df = {'y_test' : y_test, 'y_pred' : y_pred_xgb, 'y_pred_prob' : y_prob_pred_xgb}
validation_df = pd.DataFrame(data = validation_df)

# Add binning column to the dataframe
validation_df['bin_y_pred_prob'] = pd.qcut(validation_df['y_pred_prob'], q=10)
validation_df.head()

# %%
# Change x label
x_label = []
for i in range(len(validation_df['bin_y_pred_prob'].cat.categories[::-1].astype('str'))):
    x_label.append("Bin" + str(i + 1)+ "(" + validation_df['bin_y_pred_prob'].cat.categories[::-1].astype('str')[i] + ")")

# %% [markdown]
# #### Capture Rates Plot

# %%
# Plot Distribution of predicted probabilities for every bin
plt.figure(figsize=(12, 8));
sns.stripplot(validation_df.bin_y_pred_prob, validation_df.y_pred_prob, jitter = 0.15, hue = validation_df.y_test, order = validation_df['bin_y_pred_prob'].cat.categories[::-1])
plt.title("Distribution of predicted probabilities for every bin", fontsize=18)
plt.xlabel("Predicted Probability Bins", fontsize=14);
plt.ylabel("Predicted Probability", fontsize=14);
plt.xticks(np.arange(10), x_label, rotation=45);
plt.show()

# %% [markdown]
# #### Gains Table

# %%
# Aggregate the data
gains_df             = validation_df.groupby(["bin_y_pred_prob","y_test"]).agg({'y_test': ['count']})
gains_df.columns     = gains_df.columns.map(''.join)
gains_df['prob_bin'] = gains_df.index.get_level_values(0)
gains_df['y_test']   = gains_df.index.get_level_values(1)
gains_df.reset_index(drop = True, inplace = True)
gains_df

# Get infection rate and percentage infections
gains_table = gains_df.pivot(index='prob_bin', columns='y_test', values='y_testcount')
gains_table['prob_bin'] = gains_table.index
gains_table = gains_table.iloc[::-1]
gains_table['prob_bin'] = x_label
gains_table.reset_index(drop = True, inplace = True)
gains_table = gains_table[['prob_bin', 0, 1]]
gains_table.columns = ['prob_bin', "not_fraud", "fraud"]
gains_table['perc_fraud'] = gains_table['fraud']/gains_table['fraud'].sum()
gains_table['perc_not_fraud'] = gains_table['not_fraud']/gains_table['not_fraud'].sum()
gains_table['cum_perc_fraud'] = 100*(gains_table.fraud.cumsum() / gains_table.fraud.sum()) 
gains_table['cum_perc_not_fraud'] = 100*(gains_table.not_fraud.cumsum() / gains_table.not_fraud.sum()) 
gains_table


# Plot
plt.figure(figsize=(12, 8));
sns.set_style("white")
sns.pointplot(x = "prob_bin", y = "cum_perc_fraud", data = gains_table, legend = False, order=gains_table.prob_bin)
plt.xticks(rotation=45);
plt.ylabel("Fraud Rate", fontsize=14)
plt.xlabel("Prediction probability bin", fontsize=14)
plt.title("Fraud rate for every bin", fontsize=18)
plt.show()

# %% [markdown]
# Ideally the slope should be high initially and should decrease as we move further to the right. This is not really a good model.

# %%
# One big function.
def captures(y_test, y_pred, y_pred_prob):
    # Create Validation set
    validation_df = {'y_test' : y_test, 'y_pred' : y_pred, 'y_pred_prob' : y_pred_prob}
    validation_df = pd.DataFrame(data = validation_df)

    # Add binning column to the dataframe
    try:
        validation_df['bin_y_pred_prob'] = pd.qcut(validation_df['y_pred_prob'], q=10)
    except:
        validation_df['bin_y_pred_prob'] = pd.qcut(validation_df['y_pred_prob'], q=10, duplicates='drop')
    
    # Change x label and column names
    x_label = []
    for i in range(len(validation_df['bin_y_pred_prob'].cat.categories[::-1].astype('str'))):
        x_label.append("Bin" + str(i + 1)+ "(" + validation_df['bin_y_pred_prob'].cat.categories[::-1].astype('str')[i] + ")")
    
    # Plot Distribution of predicted probabilities for every bin
    plt.figure(figsize=(12, 8));
    sns.stripplot(validation_df.bin_y_pred_prob, validation_df.y_pred_prob, jitter = 0.15, hue = validation_df.y_test, order = validation_df['bin_y_pred_prob'].cat.categories[::-1])
    plt.title("Distribution of predicted probabilities for every bin", fontsize=18)
    plt.xlabel("Predicted Probability Bins", fontsize=14);
    plt.ylabel("Predicted Probability", fontsize=14);
    try:
        plt.xticks(np.arange(10), x_label, rotation=45);
    except:
        pass
    plt.show()
    
    # Aggregate the data
    gains_df             = validation_df.groupby(["bin_y_pred_prob","y_test"]).agg({'y_test': ['count']})
    gains_df.columns     = gains_df.columns.map(''.join)
    gains_df['prob_bin'] = gains_df.index.get_level_values(0)
    gains_df['y_test']   = gains_df.index.get_level_values(1)
    gains_df.reset_index(drop = True, inplace = True)
    gains_df

    # Get infection rate and percentage infections
    gains_table = gains_df.pivot(index='prob_bin', columns='y_test', values='y_testcount')
    gains_table['prob_bin'] = gains_table.index
    gains_table = gains_table.iloc[::-1]
    gains_table['prob_bin'] = x_label
    gains_table.reset_index(drop = True, inplace = True)
    gains_table = gains_table[['prob_bin', 0, 1]]
    gains_table.columns = ['prob_bin', "not_fraud", "fraud"]
    gains_table['perc_fraud'] = gains_table['fraud']/gains_table['fraud'].sum()
    gains_table['perc_not_fraud'] = gains_table['not_fraud']/gains_table['not_fraud'].sum()
    gains_table['cum_perc_fraud'] = 100*(gains_table.fraud.cumsum() / gains_table.fraud.sum()) 
    gains_table['cum_perc_not_fraud'] = 100*(gains_table.not_fraud.cumsum() / gains_table.not_fraud.sum()) 
    gains_table


    # Plot
    plt.figure(figsize=(12, 8));
    sns.set_style("white")
    sns.pointplot(x = "prob_bin", y = "cum_perc_fraud", data = gains_table, legend = False, order=gains_table.prob_bin)
    plt.xticks(rotation=45);
    plt.ylabel("Fraud Rate", fontsize=14)
    plt.xlabel("Prediction probability bin", fontsize=14)
    plt.title("Fraud rate for every bin", fontsize=18)
    plt.show()
    return gains_table

# %%
# Gains Table and Capture rates plot
captures(y_test, y_pred_xgb, y_prob_pred_xgb)

# %% [markdown]
# ## Calibration Curve

# %%
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# %%
def draw_calibration_curve(y_test, y_prob, n_bins=10):
    plt.figure(figsize=(7, 7), dpi=120)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")


    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % ("Model", ))
    ax2.hist(y_prob, range=(0, 1), bins=10, label="Model", histtype="step", lw=2)

    # Labels
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# - __Chart 1:__ X axis marks the prediction probability score. Y-axis marks the fraction of the positives.
# - __Chart 2:__ X axis marks the mean predicted value. Y-axis represents the count of records.

# %%
draw_calibration_curve(y_test, y_prob_pred_xgb, n_bins=10)

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ## Calibrate the model

# %% [markdown]
# Logistic regression

# %%
# Prediction
y_pred_xgb_test = xgb.predict(X_test)
y_prob_pred_xgb_test = xgb.predict_proba(X_test)[:, 1]

# %%
from sklearn.linear_model import LogisticRegression
X = np.array(y_prob_pred_xgb_test)
clf = LogisticRegression(random_state=0).fit(X.reshape(-1, 1), y_test)

# %%
y_prob_pred_calib = clf.predict_proba(X.reshape(-1, 1))[:, 1]
y_pred_calib      = clf.predict(X.reshape(-1, 1))

# %%
captures(y_test, y_pred_calib, y_prob_pred_calib)

# %%
draw_calibration_curve(y_test, y_prob_pred_calib, n_bins=10)

# %% [markdown]
# ### XGBoost with booster = dart

# %%
%%time
# Define the model
xgb = XGBClassifier(nthread=-1, random_state=0, booster="dart")

# Train the model
xgb.fit(X_train,y_train)

xgb

# %% [markdown]
# Let's use the model to get predictions on test dataset. We would be looking at the predicted class and predicted probability both in order to evaluate the performance of the model

# %% [markdown]
# #### Prediction

# %%
# Prediction
y_pred_xgbdart      = xgb.predict(X_test)
y_prob_pred_xgbdart = xgb.predict_proba(X_test)[:, 1]
print("Y predicted : ", y_pred_xgbdart)
print("Y probability predicted : ", y_prob_pred_xgbdart[:5])

# %% [markdown]
# #### Evaluation Metrices

# %% [markdown]
# Let's compute various evaluation metrices now
# - Accuracy Score
# - Confusion Matrix
# - Classification Report
# - AUC Score
# - Concodense Index
# - ROC curve
# - PR curve 

# %%
# Compute Evaluation Metric
compute_evaluation_metric(xgb, X_test, y_test, y_pred_xgbdart, y_prob_pred_xgbdart)

# %%
# Gains Table and Capture rates plot
captures(y_test, y_pred_xgbdart, y_prob_pred_xgbdart)

# %%
draw_calibration_curve(y_test, y_prob_pred_xgbdart, n_bins=10)

# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'> Inferences:</h3>
# </div>
# 
# - Both of the boosters are giving the similar result, maybe because both are tree based
# - The accuracy score is 0.97, AUC and concordence scores are 0.88. 
# - Recall and f1-score are very less for class True. That's because of class imbalance
# - ROC and PR curve also needs improvements
# 
# Let's look at LGBM

# %% [markdown]
# ## 15. LightGBM
# 
# LightGBM is a gradient boosting framework that uses tree based learning algorithms. 
# 
# It is designed to be distributed and efficient with the following advantages:
# 
# - Faster training speed and higher efficiency.
# - Lower memory usage.
# - Better accuracy.
# - Support of parallel and GPU learning.
# - Capable of handling large-scale data.

# %%
from lightgbm import LGBMClassifier

# %%
%%time
# Define the model
lgbc = LGBMClassifier(random_state=0, n_jobs = -1)

# Train the model
lgbc.fit(X_train,y_train)

lgbc

# %% [markdown]
# Let's use the model to get predictions on test dataset. We would be looking at the predicted class and predicted probability both in order to evaluate the performance of the model

# %%
# Prediction
y_pred_lgbc = lgbc.predict(X_test)
y_prob_pred_lgbc = lgbc.predict_proba(X_test)
y_prob_pred_lgbc = [x[1] for x in y_prob_pred_lgbc]
print("Y predicted : ",y_pred_lgbc)
print("Y probability predicted : ",y_prob_pred_lgbc[:5])

# %% [markdown]
# #### Evaluation Metrices

# %% [markdown]
# Let's compute various evaluation metrices now
# - Accuracy Score
# - Confusion Matrix
# - Classification Report
# - AUC Score
# - Concodense Index
# - ROC curve
# - PR curve 

# %%
# Compute Evaluation Metric
compute_evaluation_metric(lgbc, X_test, y_test, y_pred_lgbc, y_prob_pred_lgbc)

# %%
# Gains Table and Capture rates plot
captures(y_test, y_pred_lgbc, y_prob_pred_lgbc)

# %%
draw_calibration_curve(y_test, y_prob_pred_lgbc, n_bins=10)

# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'> Inferences:</h3>
# </div>
# 
# - With LGBM, Accuracy score is 97.7%. It's almost similar to XGBoost model
# - AUC score has imporved to 92.6 from 88.5
# - Recall and f-1 score have also improved, but it's still not upto the mark

# %% [markdown]
# ## 16. Random Forest Classifier

# %%
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

# %%
X_train.head()

# %% [markdown]
# **Impute Missing values.** Since sklearn algos are not designed to handle missing values.

# %%
from sklearn.impute import KNNImputer, SimpleImputer

# replace inf
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Impute
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer = KNNImputer(n_neighbors=3)

X_train_imputed = imputer.fit_transform(X_train)
X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
X_train_imputed.head()

# %% [markdown]
# **Build and train the Classifier**

# %%
%%time
# Define the model
rfc = RandomForestClassifier(random_state=0, n_jobs = -1)
# rfc = ExtraTreesClassifier(random_state=0, n_jobs = -1)
# rfc = AdaBoostClassifier(random_state=0)
# rfc = GradientBoostingClassifier(random_state=0)

# Train the model
rfc.fit(X_train_imputed, y_train)

rfc

# %% [markdown]
# **Predicting on test data**

# %%
# Impute X_Test before predicting
X_test_imputed = imputer.transform(X_test)

# Prediction
y_pred_rfc = rfc.predict(X_test_imputed)
y_prob_pred_rfc = rfc.predict_proba(X_test_imputed)[:, 1]

print("Y predicted : ",y_pred_rfc)
print("Y probability predicted : ",y_prob_pred_rfc[:5])

# %% [markdown]
# **Evaluation metrics**

# %%
# Compute Evaluation Metric
compute_evaluation_metric(rfc, X_test_imputed, y_test, y_pred_rfc, y_prob_pred_rfc)

# %%
# Concordance
concordance(y_test.values, y_prob_pred_rfc)

# %%
# Gains Table and Capture rates plot
captures(y_test, y_pred_rfc, y_prob_pred_rfc)

# %%
draw_calibration_curve(y_test, y_prob_pred_rfc, n_bins=10)

# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ## 17. Handling Class Imbalance

# %% [markdown]
# ### Handle Class Imbalance with Random Oversampler
# 
# Imbalanced classes are a common problem in machine learning classification where there are a disproportionate ratio of observations in each class.
# 
# Most machine learning algorithms work best when the number of samples in each class are about equal. This is because most algorithms are designed to maximize accuracy and reduce error.
# 
# - Upsample

# %%
# random over sampler
ros = RandomOverSampler()
X_train_ros, y_train_ros = ros.fit_sample(X_train_imputed, y_train)
y_train_ros.value_counts()

# %%
%%time
# Define the model
lgbc_ros = LGBMClassifier(random_state=0)

# Train the model
lgbc_ros.fit(X_train_ros,y_train_ros)

lgbc_ros

# %% [markdown]
# Let's use the model to get predictions on test dataset. We would be looking at the predicted class and predicted probability both in order to evaluate the performance of the model

# %%
# Prediction on the original test dataset
y_pred_lgbcros = lgbc_ros.predict(X_test_imputed)
y_prob_pred_lgbcros = lgbc_ros.predict_proba(X_test_imputed)[:, 1]

print("Y predicted : ",y_pred_lgbcros)
print("Y probability predicted : ",y_prob_pred_lgbcros[:5])

# %% [markdown]
# ### Evaluation Metrics

# %% [markdown]
# Let's compute various evaluation metrices now
# - Accuracy Score
# - Confusion Matrix
# - Classification Report
# - AUC Score
# - Concodense Index
# - ROC curve
# - PR curve 

# %%
# Compute Evaluation Metric
compute_evaluation_metric(lgbc_ros, X_test_imputed, y_test, y_pred_lgbcros, y_prob_pred_lgbcros)

# %%
# Gains Table and Capture rates plot
captures(y_test, y_pred_lgbcros, y_prob_pred_lgbcros)

# %%
draw_calibration_curve(y_test, y_prob_pred_lgbcros, n_bins=10)

# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'>Inferences:</h3>
# </div>
# 
# - After balancing the class, accuracy score is 0.88 and AUC score is 92.5% 
# - Accuracy has decreased as compared to the previos model, but AUC has improved
# - Additionally the recall has improved significantly at the cost of precision.
# 
# 

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ## 18. Cost Sensitive Learning with Class weights

# %% [markdown]
# The 'balanced' mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))

# %%
%%time
# Define the model
lgbc_bal = LGBMClassifier(random_state=0, class_weight='balanced')

# Train the model
lgbc_bal.fit(X_train_imputed, y_train)

lgbc_bal

# %%
# Prediction
y_pred_lgbcbal = lgbc_bal.predict(X_test)
y_prob_pred_lgbcbal = lgbc_bal.predict_proba(X_test)[:, 1]

print("Y predicted : ",y_pred_lgbcbal)
print("Y probability predicted : ",y_prob_pred_lgbcbal[:5])

# %%
# Compute Evaluation Metric
compute_evaluation_metric(lgbc_bal, X_test, y_test, y_pred_lgbcbal, y_prob_pred_lgbcbal)

# %%
# Gains Table and Capture rates plot
captures(y_test, y_pred_lgbcbal, y_prob_pred_lgbcbal)

# %%
draw_calibration_curve(y_test, y_prob_pred_lgbcbal, n_bins=10)

# %% [markdown]
# ## 19. Model Calibration

# %%
from sklearn.calibration import CalibratedClassifierCV

# %%
lgbc_bal = LGBMClassifier(random_state=0)
calibrated_clf = CalibratedClassifierCV(base_estimator=lgbc_bal, cv=3, method='sigmoid')
calibrated_clf.fit(X_train_imputed, y_train)

# %%
# Prediction
y_pred_calib = calibrated_clf.predict(X_test)
y_prob_pred_calib = calibrated_clf.predict_proba(X_test)[:, 1]

# %%
len(calibrated_clf.calibrated_classifiers_)

# %%
print("Y predicted : ", y_pred_calib)
print("Y probability predicted : ", y_prob_pred_calib[:5])

# %%
# Compute Evaluation Metric
compute_evaluation_metric(calibrated_clf, X_test_imputed, y_test, y_pred_calib, y_prob_pred_calib)

# %%
# Gains Table and Capture rates plot
captures(y_test, y_pred_calib, y_prob_pred_calib)

# %%
draw_calibration_curve(y_test, y_prob_pred_calib, n_bins=10)

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ## 20. Model Tuning
# 
# **Hyperparameter** is a parameter that governs how the algorithm trains to learn the relationships. The values are set before the learning process begins.
# 
# **Hyperparameter tuning** refers to the automatic optimization of the hyper-parameters of a ML model.

# %%
%%time
# Define the estimator
lgbmclassifier = LGBMClassifier(random_state=0)

# Define the parameters gird
param_grid = {
    'n_estimator'   : [100,200],            # default: 100
    'num_leaves'    : [256,128],            # default: 256
    'max_depth'     : [5, 8],               # default: 8 
    'learning_rate' : [0.05, 0.1],          # default: .1
    'reg_alpha'     : [0 .1, 0.5],          # default: .5
    'class_weight'  : ['balanced', None],
}

# run grid search
grid = GridSearchCV(lgbmclassifier, param_grid=param_grid, refit = True, verbose = 3, n_jobs=-1,cv = 3)
  
# fit the model for grid search 
grid.fit(X_train, y_train)

# %% [markdown]
# Get the best parameters corresponding to which you have best model

# %%
# Best parameter after hyper parameter tuning 
print(grid.best_params_) 
  
# Moel Parameters 
print(grid.best_estimator_)

lgbmclassifier = grid.best_estimator_

# %% [markdown]
# Let's use the best model to get predictions on test dataset. We would be looking at the predicted class and predicted probability both in order to evaluate the performance of the model

# %%
# Prediction using best parameters
y_grid_pred = lgbmclassifier.predict(X_test)
y_prob_grid_pred = lgbmclassifier.predict_proba(X_test)[:, 1]
print("Y predicted : ",y_grid_pred)
print("Y probability predicted : ",y_prob_grid_pred[:5])

# %% [markdown]
# ### Evaluation Metrics

# %% [markdown]
# Let's compute various evaluation metrices now
# - Accuracy Score
# - Confusion Matrix
# - Classification Report
# - AUC Score
# - Concodense Index
# - ROC curve
# - PR curve 

# %%
# Compute Evaluation Metric
compute_evaluation_metric(lgbmclassifier, X_test, y_test, y_grid_pred, y_prob_grid_pred)

# %% [markdown]
# ####  Calibration Curve

# %%
draw_calibration_curve(y_test, y_prob_grid_pred, n_bins=10)

# %% [markdown]
# ### Calibrate the model

# %%
# Calibrate
calibrated_clf = CalibratedClassifierCV(base_estimator=lgbmclassifier, cv=3)
calibrated_clf.fit(X_train, y_train)
y_pred_calib = calibrated_clf.predict(X_test)
y_prob_pred_calib = calibrated_clf.predict_proba(X_test)[:, 1]

# %%
draw_calibration_curve(y_test, y_prob_pred_calib, n_bins=10)

# %%
# Compute Evaluation Metric
compute_evaluation_metric(calibrated_clf, X_test, y_test, y_pred_calib, y_prob_pred_calib)

# %%
# Gains Table and Capture rates plot
captures(y_test, y_pred_calib, y_prob_pred_calib)

# %%


# %%


# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'>Inferences:</h3>
# </div>
# 
# - Accuracy score is 0.91
# - AUC score and Concordance index are 0.97, which are the best so far
# - Classfication report is also balanced between both the classes
# - ROC curve and PR are the best so far

# %% [markdown]
# Hence we can freeze the model.

# %% [markdown]
# ## 21. Feature Importance
# 
# Feature importance refers to techniques that assign a score to input features based on how useful they are at predicting a target variable.

# %%
lgbmclassifier = grid.best_estimator_

# %%
lgbmclassifier.feature_importances_ 

# %%
feature_importance_df = pd.DataFrame({'feature' : X_train.columns, 'importance' : lgbmclassifier.feature_importances_ })

# %%
feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)
feature_importance_df = feature_importance_df.iloc[:30,:]
feature_importance_df

# %%
plt.figure(figsize=(16, 12));
sns.barplot(x="importance", y="feature", data=feature_importance_df.sort_values(by="importance", ascending=False));
plt.title('LGB Features');

# %% [markdown]
# <div class="alert alert-info" style="padding:0px 10px; border-radius:5px;"><h3 style='margin:10px 5px'> Inferences:</h3>
# </div>
# 
# - card1 is contributing the most in predicting if a transaction is fraud or not
# - card2, addr1, C13, P_emaildomain, C1 etc are some of the most important features in predicting the fraud
# - Certain card types, addresses and emails are at high risk of fraud, so there is a need to monitor these carefully

# %% [markdown]
# ## 22. Partial Dependence and Individual Conditional Expectations (ICE)

# %%
## pdp plots
from sklearn.inspection import partial_dependence, plot_partial_dependence
from sklearn.utils import validation

# %% [markdown]
# Fit the model

# %%
lgbmclassifier.fit(X_train, y_train)
lgbmclassifier.dummy_ = "dummy"

validation.check_is_fitted(estimator=lgbmclassifier)

# %% [markdown]
# __Plot Partial Dependence__

# %%
fig = plt.figure(figsize=(16, 12))
plot_partial_dependence(lgbmclassifier, X, ['card2'])
plt.show()

# %% [markdown]
# __Individual Conditional Expectation (ICE) Plot - card2__

# %%
plot_partial_dependence(lgbmclassifier, X, ['card2'], kind='both')

# %% [markdown]
# __Partial Dependence  and ICE Plot - C13__

# %%
fig = plt.figure(figsize=(16, 12))
plot_partial_dependence(lgbmclassifier, X, ['C13'], kind='both')
plt.show()

# %%
fig = plt.figure(figsize=(16, 12))
plot_partial_dependence(lgbmclassifier, X, ['C13'])
plt.show()

# %% [markdown]
# ## 23. SHAP Values

# %% [markdown]
# SHAP values is used to reverse engineer the output of the prediction model and quantify the contribution of each predictor for a given prediction.

# %%
import shap
shap_model = shap.TreeExplainer(lgbmclassifier)
shap_values = shap_model.shap_values(X_train)

# %% [markdown]
# You can make a partial dependence plot using `shap.dependence_plot`. This shows the relationship between the feature and the Y. This also automatically includes another feature that your feature interacts frequently with.
# 
# 

# %%
# card2
shap.dependence_plot("card2", shap_values[0], X_train)

# %%
# card3
shap.dependence_plot("card3", shap_values[0], X_train)

# %% [markdown]
# **Explain a single observation.**

# %%
shap.initjs()  # needed to show viz
shap.force_plot(shap_model.expected_value[1], shap_values[1][14], X_train.iloc[14, :])

# %% [markdown]
# Add link = "logit"

# %%
shap.initjs()  # needed to show viz
shap.force_plot(shap_model.expected_value[1], shap_values[1][14], X_train.iloc[14, :], link='logit')

# %%
y_pred_calib_tr[14]

# %%
# compute SHAP values
explainer = shap.Explainer(lgbmclassifier, X_train) # , link=shap.links.logit)
shap_values_waterfall = explainer(X_train[:100])

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values_waterfall[0])

# %%


# %%


# %%


# %%


# %% [markdown]
# ## Final Words
# 
# 0. __Problem Understanding__
# 1. __Data Overview__
# 2. __Optimize memory__
# 3. __Data Stats and Exploratory Data Analysis__
# 4. __Significance Tests__
# 5. __Feature Engineering, Encoding, PCA__
# 6. __XGBoost__
# 7. __Evaluation Metrics for Fraud classification__
# 8. __Calibration__
# 9. __Handling class imbalance with oversampling__
# 10. __Cost sensitive Learning__
# 11. __Model Tuning__
# 12. __Feature Importance__
# 13. __Partial Dependence and ICE__
# 14. __SHAP values__
# 
# The model has been trained and tested, so now you can use it to predict if any transaction would be fraud or not. 
# 
# One very important thing to note is that you need monitor the performance of the model as you gather new data. If it deteriorates, re-train the model with new training data and use it for making detecting fraud transactions.

# %%


# %%


# %%


