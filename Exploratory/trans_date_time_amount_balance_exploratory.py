from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df['Is_Fraud'] = df['Is_Fraud'].replace({0: 'No', 1: 'Yes'})

    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
    df['Year'] = df['Transaction_Date'].dt.year
    df['Month'] = df['Transaction_Date'].dt.month
    df['Day'] = df['Transaction_Date'].dt.day
    df['DayOfWeek'] = df['Transaction_Date'].dt.dayofweek  # Monday is zero, Sunday is 6
    df['DayName'] = df['Transaction_Date'].dt.day_name()  
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
# fraud per day
daily_summary = CLEANED.groupby('Transaction_Date').agg(
    total_transactions=('Transaction_Date', 'count'),
    fraud_transactions=('Is_Fraud', lambda x: (x == 'Yes').sum())
).reset_index()
daily_summary['fraud_rate'] = (
    daily_summary['fraud_transactions'] 
    / daily_summary['total_transactions']
)

daily_summary = daily_summary[daily_summary['total_transactions'] > 0]

daily_summary['fraud_rate_7d'] = (
    daily_summary['fraud_rate']
    .rolling(window=7, center=True)
    .mean()
)

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

X_amt_hr = df2[['LogAmt','Hour']]
auc_amt_hr = cross_val_score(
    LogisticRegression(max_iter=500), 
    X_amt_hr, y, cv=5, scoring='roc_auc'
).mean()

print(f"AUC (amount only):  {auc_amt:.3f}") #! Note; the AUC is like identical between these two, so hour isn't really a meaningful feature in this context.
print(f"AUC (amount+hour): {auc_amt_hr:.3f}")

df2 = CLEANED.copy()
df2['LogAmt'] = np.log1p(df2['Transaction_Amount'])

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

mask_w = df2['Transaction_Type'].isin(['Withdrawal','Transfer'])
df2.loc[mask_w,   'Bal_Pre'] = df2.loc[mask_w,   'Account_Balance'] + df2.loc[mask_w,   'Transaction_Amount']

mask_d = df2['Transaction_Type']=='Deposit'
df2.loc[mask_d,   'Bal_Pre'] = df2.loc[mask_d,   'Account_Balance'] - df2.loc[mask_d,   'Transaction_Amount']
df2['Amt_to_PreBal'] = df2['Transaction_Amount'] / (df2['Bal_Pre'] + 1e-6)
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

