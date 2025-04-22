import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier


xg_bank_fraud_df = pd.read_csv('Cleaned_Fraud.csv')


# One-hot encoding: Gender, Account_Type, Transaction_Type, Merchant_Category

xg_bank_fraud_df['Gender'] = xg_bank_fraud_df['Gender'].map({
    'Male' : 1, 'Female': 0
})

xg_bank_fraud_df['Account_Type'] = xg_bank_fraud_df['Account_Type'].map({
    'Savings' : 0, 'Business' : 1, 'Checking' : 2, 
})

xg_bank_fraud_df['Transaction_Type'] = xg_bank_fraud_df['Transaction_Type'].map({
    'Transfer' : 0, 'Bill Payment' : 1, 'Debit' : 2, 'Withdrawal' : 3, 'Credit' : 4,
})

xg_bank_fraud_df['Merchant_Category'] =  xg_bank_fraud_df['Merchant_Category'].map({
    'Restaurant' : 0, 'Groceries' : 1, 'Entertainment' : 2, 'Health' : 3, 'Clothing' : 4, 'Electronics' : 5, 
})



# Frequency encoding: State, City, Bank_Branch, Transaction_Device, Device_Type
# Included this type of encoding because of the number of unique values is a lot
for column in ['State', 'City', 'Bank_Branch', 'Transaction_Device', 'Device_Type']:
    freq_map = xg_bank_fraud_df[column].value_counts().to_dict()
    xg_bank_fraud_df[column] = xg_bank_fraud_df[column].map(freq_map)


# Converting Date and Time Columns into useable columns
xg_bank_fraud_df['Transaction_Date'] = pd.to_datetime(xg_bank_fraud_df['Transaction_Date'])

xg_bank_fraud_df['Year'] = xg_bank_fraud_df['Transaction_Date'].dt.year
xg_bank_fraud_df['Month'] = xg_bank_fraud_df['Transaction_Date'].dt.month
xg_bank_fraud_df['Day'] = xg_bank_fraud_df['Transaction_Date'].dt.day
xg_bank_fraud_df['Weekday'] = xg_bank_fraud_df['Transaction_Date'].dt.weekday

xg_bank_fraud_df['Transaction_Time'] = pd.to_datetime(xg_bank_fraud_df['Transaction_Time'], format='%H:%M:%S', errors='coerce')
xg_bank_fraud_df['Hour'] = xg_bank_fraud_df['Transaction_Time'].dt.hour
xg_bank_fraud_df['Minute'] = xg_bank_fraud_df['Transaction_Time'].dt.minute

# Csv to see if it was made correctly
#xg_bank_fraud_df.to_csv('testing.csv', index=False)




# Save Is_Fraud column, remove so model doesn't self predict
y = xg_bank_fraud_df['Is_Fraud']
x = xg_bank_fraud_df.drop(['Is_Fraud', 'Transaction_Location', 'Transaction_Date', 'Transaction_Time'], axis=1)


#Split data 80/20
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3870)

# make and fit the model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)


# check predictions
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
