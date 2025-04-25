from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    # Is-Fraud as a word feature
    df['Is_Fraud'] = df['Is_Fraud'].replace({0: 'No', 1: 'Yes'})

    # Date manipulation for easier comparisons
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
    df['Year'] = df['Transaction_Date'].dt.year
    df['Month'] = df['Transaction_Date'].dt.month
    df['Day'] = df['Transaction_Date'].dt.day
    df['DayOfWeek'] = df['Transaction_Date'].dt.dayofweek  # Monday is zero, Sunday is 6
    df['DayName'] = df['Transaction_Date'].dt.day_name()  
    # Time manipulation, for easy comparisons (datetime package)
    df['Transaction_Time'] = pd.to_datetime(df['Transaction_Time'], format='%H:%M:%S').dt.time
    df['Hour'] = pd.to_datetime(df['Transaction_Time'], format='%H:%M:%S').dt.hour
    return df
CLEANED: pd.DataFrame = pd.read_csv('Cleaned_Fraud.csv')
CLEANED = transform_features(df=CLEANED)

##############################
## Transaction time & Fraud ##
##############################
def plot_trans_time_vs_fraud():
    fraud_by_transaction_time = CLEANED.groupby(['Hour', 'Is_Fraud']).size().unstack(fill_value=0)  
    fraud_by_transaction_time.plot(
        kind='bar', 
        stacked=True,
        figsize=(10, 6)
    )
    plt.title("Total Transactions vs Fraudulent Transactions by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Count of Transactions")
    plt.legend(title="Is Fraud")
    plt.tight_layout()
    #plt.show()
    fraud_only = CLEANED[CLEANED['Is_Fraud'] == 'Yes']
    mean_fraud_hour = fraud_only['Hour'].mean()

    print("Average hour (fraud):", mean_fraud_hour)

    mean_all_trans = CLEANED['Hour'].mean()
    print("Avg time for all transactions:", mean_all_trans)
    df_fraud = CLEANED[CLEANED['Is_Fraud'] == 'Yes']

    fraud_hour_std = df_fraud['Hour'].std() 

    print(f"Standard Deviation of Fraud Hours: {fraud_hour_std:.2f}")

    df_non_fraud = CLEANED[CLEANED['Is_Fraud'] == 'No']
    non_fraud_hour_std = df_non_fraud['Hour'].std()

    print(f"Std of Fraud Hours: {fraud_hour_std:.2f}")
    print(f"Std of Non-Fraud Hours: {non_fraud_hour_std:.2f}")

    std_df = pd.DataFrame({
        'Type': ['Fraud', 'Non-Fraud'],
        'StdDev': [fraud_hour_std, non_fraud_hour_std]
    })
    plt.figure(figsize=(6, 4))
    plt.bar(std_df['Type'], std_df['StdDev'])
    plt.title("Standard Deviation of Transaction Hours: Fraud vs Non-Fraud")
    plt.xlabel("Transaction Type")
    plt.ylabel("Standard Deviation of Hour")
    plt.ylim(0, max(std_df['StdDev']) * 1.2)  
    plt.show()
#! Fraud times alone don't say much about chance of fraud.

##############################
## Transaction date & Fraud ##
##############################
# fraud per day# 1) Build daily_summary as before
daily_summary = CLEANED.groupby('Transaction_Date').agg(
    total_transactions=('Transaction_Date', 'count'),
    fraud_transactions=('Is_Fraud', lambda x: (x == 'Yes').sum())
).reset_index()
daily_summary['fraud_rate'] = (
    daily_summary['fraud_transactions'] 
    / daily_summary['total_transactions']
)

# 2) Drop days with zero transactions
daily_summary = daily_summary[daily_summary['total_transactions'] > 0]

# 3) Compute 7-day centered rolling average
daily_summary['fraud_rate_7d'] = (
    daily_summary['fraud_rate']
    .rolling(window=7, center=True)
    .mean()
)

# 4) Plot
plt.figure(figsize=(12, 6))

# raw daily rate in light grey
sns.lineplot(
    data=daily_summary,
    x='Transaction_Date',
    y='fraud_rate',
    color='lightgray',
    linewidth=1,
    label='Daily Fraud Rate',
    marker=None
)

# smoothed 7-day average in red
sns.lineplot(
    data=daily_summary,
    x='Transaction_Date',
    y='fraud_rate_7d',
    color='red',
    linewidth=2,
    label='7-Day Rolling Avg'
)

plt.xlabel("Transaction Date")
plt.ylabel("Fraud Rate")
plt.title("Daily Fraud Rate with 7-Day Rolling Average")
plt.legend()
plt.tight_layout()
plt.show()
#############################
## Time & Amount vs. Fraud ##
#############################

df2 = CLEANED.copy()
df2['LogAmt'] = np.log1p(df2['Transaction_Amount'])
y = df2['Is_Fraud'].map({'No':0,'Yes':1})

X_amt = df2[['LogAmt']]
auc_amt = cross_val_score(
    LogisticRegression(max_iter=500), 
    X_amt, y, cv=5, scoring='roc_auc'
).mean()

# 2) Amount + Hour
X_amt_hr = df2[['LogAmt','Hour']]
auc_amt_hr = cross_val_score(
    LogisticRegression(max_iter=500), 
    X_amt_hr, y, cv=5, scoring='roc_auc'
).mean()

print(f"AUC (amount only):  {auc_amt:.3f}") #! Note; the AUC is like identical between these two, so hour isn't really a meaningful feature in this context.
print(f"AUC (amount+hour): {auc_amt_hr:.3f}")
"""

df2 = CLEANED.copy()
df2['LogAmt'] = np.log1p(df2['Transaction_Amount'])

# split out the two groups
"""
"""
The following demonstrates a KDE for fraud and non-fraud transactions based on the amount.
The curves are pretty aligned, with a large overlap. However, the right tail of the fraud curve extends farther slightly,
meaning that at extreme ends, (ie, for large transaction amounts), the probability of a transaction being fraudulent is much higher than non-fraudulent.
However, there is still mostly overlap; so this only identifies a very small percentage of overall fraudulent transactions; most fraud transaction amounts
still lie within the same ranges as normal transactions




"""

#! This plots the fraud rate in relation to what percentage of the balance is drained from a withdrawal/transfer. 
#! That is, if my balance is 100$, and a withdrawal of 99$ occurs, is that more likely to be fraudulent than a withdrawal of 50$? 
#! -> Generally, no, it isnt a good indicator.

df2 = CLEANED.copy()

# for withdrawals/transfers, the new balance = old_balance – amount
# so old_balance = new_balance + amount
mask_w = df2['Transaction_Type'].isin(['Withdrawal','Transfer'])
df2.loc[mask_w,   'Bal_Pre'] = df2.loc[mask_w,   'Account_Balance'] + df2.loc[mask_w,   'Transaction_Amount']

# for deposits,    new_balance = old_balance + amount
# so         old_balance = new_balance – amount
mask_d = df2['Transaction_Type']=='Deposit'
df2.loc[mask_d,   'Bal_Pre'] = df2.loc[mask_d,   'Account_Balance'] - df2.loc[mask_d,   'Transaction_Amount']
df2['Amt_to_PreBal'] = df2['Transaction_Amount'] / (df2['Bal_Pre'] + 1e-6)
# (if you have other types, handle them similarly or drop those rows)
plt.figure(figsize=(8,6))
sns.violinplot(
    data=df2, 
    x='Is_Fraud', 
    y='Amt_to_PreBal',
    inner='quartile',
    scale='width'
)
plt.ylim(0,1)   
plt.title('Withdrawal Amount as Fraction of Pre-Transaction Balance')
plt.ylabel('Amount ÷ Balance Before Txn')
plt.xlabel('Is it Fraudulent?')
plt.show()
"""


# 1) Load & label
df = pd.read_csv('Cleaned_Fraud.csv', parse_dates=['Transaction_Date'])
df['y'] = df['Is_Fraud'].astype(int)

# 2) Feature‐engineer date+time
df['Hour'] = pd.to_datetime(df['Transaction_Time'], format='%H:%M:%S').dt.hour
df['Dow']  = df['Transaction_Date'].dt.dayofweek   # 0=Mon…6=Sun
df['Mon']  = df['Transaction_Date'].dt.month

# 3) Amount & balance features
df['LogAmt']      = np.log1p(df['Transaction_Amount'])
df['LogBal']      = np.log1p(df['Account_Balance'])
mask = df['Transaction_Type'].isin(['Withdrawal','Transfer'])
df.loc[mask, 'Bal_Pre']    = df.loc[mask, 'Account_Balance'] + df.loc[mask, 'Transaction_Amount']
df['Amt_to_PreBal'] = df['Transaction_Amount']/ (df['Bal_Pre'] + 1e-6)

# 4) Encode everything in one go
X = pd.get_dummies(
    df[[
      'LogAmt','LogBal','Amt_to_PreBal','Hour','Dow','Mon'
    ]],
    columns=['Dow','Mon'],         # only these need dummies
    prefix=['dow','mon'],
    drop_first=True
)
y = df['y']
mask = df['y'].notna()
X = X.loc[mask]
y = df['y'].loc[mask]
# 5) Validate joint signal
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
print("ROC-AUC:", cross_val_score(rf, X, y, cv=5, scoring='roc_auc').mean())
"""