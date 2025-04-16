from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


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
"""
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
"""
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
"""
plt.figure(figsize=(6, 4))
plt.bar(std_df['Type'], std_df['StdDev'])
plt.title("Standard Deviation of Transaction Hours: Fraud vs Non-Fraud")
plt.xlabel("Transaction Type")
plt.ylabel("Standard Deviation of Hour")
plt.ylim(0, max(std_df['StdDev']) * 1.2)  
#plt.show()
"""
#! Fraud times alone don't say much about chance of fraud.

##############################
## Transaction date & Fraud ##
##############################
# fraud per day
daily_summary = CLEANED.groupby('Transaction_Date').agg(
    total_transactions=('Transaction_Date', 'count'),
    fraud_transactions=('Is_Fraud', lambda x: (x == 'Yes').sum())
).reset_index()
daily_summary['fraud_rate'] = daily_summary['fraud_transactions'] / daily_summary['total_transactions']

fig, ax1 = plt.subplots(figsize=(12, 6))

sns.lineplot(
    data=daily_summary,
    x='Transaction_Date',
    y='fraud_rate',
    marker='o',
    ax=ax1,
    label="Fraud Rate",
    color='red'
)
ax1.set_xlabel("Transaction Date")
ax1.set_ylabel("Fraud Rate", color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
sns.lineplot(
    data=daily_summary,
    x='Transaction_Date',
    y='total_transactions',
    marker='o',
    ax=ax2,
    label="Total Transactions",
    color='blue'
)
ax2.set_ylabel("Total Transactions", color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title("Daily Fraud Rate and Total Transactions Over Time")
plt.show()