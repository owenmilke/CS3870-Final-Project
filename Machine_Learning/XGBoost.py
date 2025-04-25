import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, fbeta_score
from sklearn.utils import resample
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load in data
xg_bank_fraud_df = pd.read_csv('Bank_Transaction_Fraud_Detection.csv')

# ---------------------- XG Specific Cleaning ----------------------
# Getting rid of useless columns
xg_bank_fraud_df.drop(['Customer_Name', 'Transaction_ID', 'Transaction_Currency', 'Transaction_Location', 'Customer_Contact', 'Customer_Email', 'Customer_ID', 'Merchant_ID'], axis=1, inplace=True)

# Format time and date to be usable
xg_bank_fraud_df['Transaction_Date'] = pd.to_datetime(xg_bank_fraud_df['Transaction_Date'], format='%d-%m-%Y', dayfirst=True)
xg_bank_fraud_df['Transaction_Time'] = pd.to_datetime(xg_bank_fraud_df['Transaction_Time'], format='%H:%M:%S', errors='coerce')


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

# Transaction_Date
xg_bank_fraud_df['Year'] = xg_bank_fraud_df['Transaction_Date'].dt.year
xg_bank_fraud_df['Month'] = xg_bank_fraud_df['Transaction_Date'].dt.month
xg_bank_fraud_df['Day'] = xg_bank_fraud_df['Transaction_Date'].dt.day
xg_bank_fraud_df['Weekday'] = xg_bank_fraud_df['Transaction_Date'].dt.weekday

# Transaction_Time
xg_bank_fraud_df['Hour'] = xg_bank_fraud_df['Transaction_Time'].dt.hour
xg_bank_fraud_df['Minute'] = xg_bank_fraud_df['Transaction_Time'].dt.minute

# Transaction_Amount - Nothing changed


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


# ---------------------- Setup Data for Model ----------------------
# Save Is_Fraud column, remove so model doesn't self predict
y = xg_bank_fraud_df['Is_Fraud']
x = xg_bank_fraud_df.drop(
    ['Is_Fraud', 
     'Transaction_Date', 
     'Transaction_Time', 
     'Transaction_Description'], 
     axis=1)


#Split data 80/20
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3870, stratify=y)


# ---------------------- Oversampling Frauds ----------------------
# Re-assemble training data
train_df = pd.concat([X_train, y_train.rename('Is_Fraud')], axis=1)

# Separate majority/minority
df_majority = train_df[train_df.Is_Fraud == 0]
df_minority = train_df[train_df.Is_Fraud == 1]

# Upsample minority (fraud) to match majority
df_minority_upsampled = resample(
    df_minority,
    replace=True,          
    n_samples=len(df_majority), 
    random_state=3870
)

# Combine back into a balanced DataFrame and shuffle
train_balanced = pd.concat([df_majority, df_minority_upsampled])
train_balanced = train_balanced.sample(frac=1, random_state=3870)

# Split back into X/y
X_train, y_train = (
    train_balanced.drop('Is_Fraud', axis=1),
    train_balanced['Is_Fraud']
)

#print(x)
#x.to_csv('test_model.csv', index=False)

#print(f"y train vals: {y_train.value_counts()}")


# ---------------------- Make and fit the models ----------------------
# Simple model to get most important features
first_model = XGBClassifier(
    n_estimators=100,
    max_depth=5, 
    random_state=3870,
    scale_pos_weight=1,
)

first_model.fit(X_train, y_train)

# Find the most important features
importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': first_model.feature_importances_
}).sort_values('Importance', ascending=False)
top_features = importance.head(12)['Feature'].values

# Re-set the X_train and test
X_train_filtered = X_train[top_features]
X_test_filtered = X_test[top_features]

# More compilcated model
final_model = XGBClassifier(
    n_estimators=200,
    max_depth=12,
    learning_rate=0.1,
    objective='binary:logistic',
    #eval_metric='aucpr',  # Better for imbalanced data
    subsample=0.7,
    #colsample_bytree=0.8,
    #reg_alpha=0.1,  # L1 regularization
    #reg_lambda=0.1,  # L2 regularization
    scale_pos_weight=1, 
    random_state=3870
)
final_model.fit(X_train_filtered, y_train)


# Find the optimal threshold for precision/recall curve
y_prob = final_model.predict_proba(X_test_filtered)[:, 1]

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

precision, recall, threshold = precision_recall_curve(y_test, y_prob)
plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(" Curve")
plt.show()

valid = np.where(recall[:-1] >= 0.3)[0]
best_idx = valid[np.argmax(precision[valid])]
balanced_threshold = threshold[best_idx]




# Prediction w/ balanced_threshold
y_pred_optimized = (y_prob >= balanced_threshold).astype(int)
f2 = fbeta_score(y_test, y_pred_optimized, beta=2)


# ---------------------- RESULTS ----------------------
# Evaluate with optimized threshold
print(f"Balanced Threshold: {balanced_threshold:.4f}")
print(f"F2 Score: {f2:.3f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred_optimized, zero_division=0))
print("Confusion Matrix:")
print(pd.crosstab(y_test, y_pred_optimized, 
                 rownames=['Actual'], 
                 colnames=['Predicted'], 
                 margins=True))


plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall curve")
plt.xlim(-0.001, 0.01)
plt.show()

print("Feature importance: ")
print("\nTop Features:")
print(importance)