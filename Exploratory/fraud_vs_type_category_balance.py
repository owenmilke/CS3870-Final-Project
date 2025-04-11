import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Cleaned_Fraud.csv")
df['Is_Fraud'] = df['Is_Fraud'].replace({0: 'No', 1: 'Yes'})

# Set up figure
fig, axes = plt.subplots(1, 3, figsize=(24, 6))

# Correlation between transaction type and fraud
tx_counts = df.groupby(['Transaction_Type', 'Is_Fraud']).size().unstack(fill_value=0)
tx_pct = tx_counts.div(tx_counts.sum(axis=1), axis=0)
tx_pct.plot.bar(
    stacked=True,
    title="Percentage of Fraud by Transaction Type",
    ylabel='',
    xlabel='',
    ax=axes[0],
    rot=0,
)
for container in axes[0].containers:
    axes[0].bar_label(container, 
                labels=[f'{x:.2%}' if x > 0.01 else '' for x in container.datavalues],
                label_type='center',
                fontsize=10)
axes[0].legend(title="Is Fraud")
axes[0].set_yticks([])

# Correlation between merchant category and fraud
sns.countplot(
    data=df,
    x="Merchant_Category",
    hue="Is_Fraud",
    ax=axes[1]
)
axes[1].set_title("Fraud Count by Merchant Category")
axes[1].set_xlabel('')
axes[1].set_ylabel('')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title="Is Fraud", loc='lower left')

# Correlation between account balance and fraud
sns.violinplot(
    data=df,
    x="Is_Fraud",
    y="Account_Balance",
    hue="Is_Fraud",
    split=True,
    ax=axes[2],
    legend=False
)
axes[2].set_title("Account Balance Distribution by Fraud")
axes[2].set_xlabel('')
axes[2].set_ylabel('')

plt.tight_layout()
plt.show()