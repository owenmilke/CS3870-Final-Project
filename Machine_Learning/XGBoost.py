import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, fbeta_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Load in data
xg_bank_fraud_df = pd.read_csv('Bank_Transaction_Fraud_Detection.csv')

# ---------------------- XGB Data Cleaning ----------------------
# Format time and date to be usable
xg_bank_fraud_df['Transaction_Date'] = pd.to_datetime(xg_bank_fraud_df['Transaction_Date'], format='%d-%m-%Y', dayfirst=True)
xg_bank_fraud_df['Transaction_Time'] = pd.to_datetime(xg_bank_fraud_df['Transaction_Time'], format='%H:%M:%S', errors='coerce')

# Transaction_Date
xg_bank_fraud_df['Year'] = xg_bank_fraud_df['Transaction_Date'].dt.year
xg_bank_fraud_df['Month'] = xg_bank_fraud_df['Transaction_Date'].dt.month
xg_bank_fraud_df['Day'] = xg_bank_fraud_df['Transaction_Date'].dt.day
xg_bank_fraud_df['Weekday'] = xg_bank_fraud_df['Transaction_Date'].dt.weekday

# Transaction_Time
xg_bank_fraud_df['Hour'] = xg_bank_fraud_df['Transaction_Time'].dt.hour
xg_bank_fraud_df['Minute'] = xg_bank_fraud_df['Transaction_Time'].dt.minute

# Gender
xg_bank_fraud_df['Gender'] = xg_bank_fraud_df['Gender'].map({
    'Male' : 1, 'Female': 0
})

# Age
age_bins = [0, 25, 40, 60, 120]
labels = ['less25','26-40','41-60','60plus']
xg_bank_fraud_df['Age_Group'] = pd.cut(xg_bank_fraud_df['Age'], bins=age_bins, labels=labels)
xg_bank_fraud_df = pd.get_dummies(xg_bank_fraud_df, columns=['Age_Group'], prefix='AgeGrp')

# Account_Type
xg_bank_fraud_df['Account_Type'] = xg_bank_fraud_df['Account_Type'].map({
    'Savings' : 0, 'Business' : 1, 'Checking' : 2, 
})

# Transaction_Type
xg_bank_fraud_df['Transaction_Type'] = xg_bank_fraud_df['Transaction_Type'].map({
    'Transfer' : 0, 'Bill Payment' : 1, 'Debit' : 2, 'Withdrawal' : 3, 'Credit' : 4,
})

# Merchant_Category
xg_bank_fraud_df['Merchant_Category'] =  xg_bank_fraud_df['Merchant_Category'].map({
    'Restaurant' : 0, 'Groceries' : 1, 'Entertainment' : 2, 'Health' : 3, 'Clothing' : 4, 'Electronics' : 5, 
})

# Account_Balance
xg_bank_fraud_df['Amt_Trans_to_Bal_Ratio'] = xg_bank_fraud_df['Transaction_Amount'] / (xg_bank_fraud_df['Account_Balance'] + 1)

# Transaction_Description
keyword_flags = [
    'ATM','Bitcoin','Cryptocurrency','Online','Refund', 'POS','Transfer',
    'Subscription','Electronics','Gift','Charity','Rental','Taxi', 'Penalty', 'Pharmacy', 'Luxury'
    ]

for kw in keyword_flags:
    col_name = f'Desc_Has_{kw}'
    # Check if keyword exists (case-insensitive), handle NaN values
    xg_bank_fraud_df[col_name] = (
        xg_bank_fraud_df['Transaction_Description']
        .str.contains(kw, case=False, regex=False)
        .fillna(False)
        .astype(int)
    )

# Frequency encoding: State, City, Bank_Branch, Transaction_Device, Device_Type
# Included this type of encoding because of the number of unique values is a lot
for column in ['State', 'City', 'Bank_Branch', 'Transaction_Device', 'Device_Type']:
    freq_map = xg_bank_fraud_df[column].value_counts().to_dict()
    xg_bank_fraud_df[column] = xg_bank_fraud_df[column].map(freq_map)

# Getting rid of non categorical/informative variabless
xg_bank_fraud_df.drop(['Customer_Name', 'Transaction_ID', 'Transaction_Currency', 
                       'Transaction_Location', 'Customer_Contact', 'Customer_Email', 
                       'Customer_ID', 'Merchant_ID', 'Transaction_Date', 'Transaction_Time', 
                       'Transaction_Description'], 
                       axis=1, 
                       inplace=True)


# ---------------------- Setup Data for modeling ----------------------
# Is_Fraud is what we want to predict
y = xg_bank_fraud_df['Is_Fraud']
x = xg_bank_fraud_df.drop(['Is_Fraud'], axis=1)

# Found StandardScaler to standardize all numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# Used smote for resampling, giving new synthetic rows to help even out fraud/non-fraud
smote = SMOTE(random_state=3870)
X_resampled, y_resampled = smote.fit_resample(x, y)

#Split data 80/20
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=3870, stratify=y_resampled)

# ---------------------- Make and fit the models ----------------------
# Initialize model
xg_model = XGBClassifier(
    n_estimators=300,
    max_depth=12,
    learning_rate=0.1,
    eval_metric='logloss',
    subsample=0.7,
    scale_pos_weight=1, 
    random_state=3870
)

# Fit model
xg_model.fit(X_train, y_train)

# Find the optimal threshold for precision/recall curve
y_prob = xg_model.predict_proba(X_test)[:, 1]

# Plot ROC Curve
fpr, tpr, roc_th = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Plot Precision/Recall curve
precision, recall, threshold = precision_recall_curve(y_test, y_prob)
plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(" Curve")
plt.show()

valid = np.where(recall[:-1] >= 0.8)[0]
best_idx = valid[np.argmax(precision[valid])]
threshold = threshold[best_idx]

# Prediction w/ balanced_threshold
y_pred_optimized = (y_prob >= threshold).astype(int)


# ---------------------- RESULTS ----------------------
print("Classification Report:")
print(classification_report(y_test, y_pred_optimized, zero_division=0))
print("Confusion Matrix:")
print(pd.crosstab(y_test, y_pred_optimized, 
                 rownames=['Actual'], 
                 colnames=['Predicted'], 
                 margins=True))

