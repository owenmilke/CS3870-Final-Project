import pandas as pd

# Read in full csv
fraud_df = pd.read_csv("Bank_Transaction_Fraud_Detection.csv")

# Drop unnecessary columns
fraud_df = fraud_df.drop(
                        ['Customer_ID', 'Customer_Name', 'Transaction_ID',
                         'Merchant_ID', 'Transaction_Description', 'Customer_Email', 'Customer_Contact', 'Transaction_Currency'], 
                        axis=1)

# Switch Is_Fraud's location to first column
is_fraud = fraud_df.pop('Is_Fraud')
fraud_df.insert(0, 'Is_Fraud', is_fraud)

# Standardize Date
fraud_df['Transaction_Date'] = pd.to_datetime(fraud_df['Transaction_Date'], dayfirst=True, errors='coerce').dt.date
# Standardize Time
fraud_df['Transaction_Time'] = pd.to_datetime(fraud_df['Transaction_Time'], format='%H:%M:%S', errors='coerce').dt.time

# Make into new csv
fraud_df.to_csv('Cleaned_Fraud.csv', index=False)

