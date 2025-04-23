import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.express as px

#Note --> Im doing City, Bank_Branch, Account_Type, Device_Type
#reading in the cleaned dataset [thanks team!]
data = pd.read_csv("Cleaned_Fraud.csv")

#dropping unnecessary columns
data = data.drop(["Gender", "Age", "State", "Transaction_Date", "Transaction_Time", "Transaction_Amount", "Merchant_Category", "Account_Balance",
         "Transaction_Device", "Transaction_Location"], axis = 1)

#grouping the data by the factor [being City, Bank_Branch, Account_Type and Device Type
fraudFactors = ['City','Bank_Branch','Account_Type','Device_Type']

#dictionary, making the column names more readable
factorDict = {"City" : "City",
          "Bank_Branch" : "Bank Branch",
          "Account_Type" : "Account Type",
          "Device_Type": "Device Type"}

#constructing the plot, which consists of 4 plots
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
axes = axes.flatten()

#iterating through the factors
for i, factor in enumerate(fraudFactors):
    if factor in ['City', 'Bank_Branch']:
        #If the factor is City or Bank Branch, only keeping the five most pertinent [frequent] ones, for ease in visualization
        topValues = data[factor].value_counts().nlargest(5).index
        filter = data[data[factor].isin(topValues)]
    else:
        filter = data

    #grouping by Is_Fraud
    group = (
        filter.groupby([factor, 'Is_Fraud'])
        .size()
        .unstack(fill_value=0)
    )

    #getting the percentage
    grouping_percent = group.div(group.sum(axis=1), axis=0) * 100

    #forming the bar plots --> Green is not fraud, and red is Fraud
    ax = grouping_percent.plot(
        kind='bar',
        stacked=True,
        ax=axes[i],
        color=['green', 'red']  
    )

    #formatting the output to make it more readable
    for container in ax.containers:
        labels = [f'{v:.2f}%' if v > 0 else '' for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='center', fontsize=9, color='white', fontweight='bold')

    #changing the axis to the dictionary-dictated column
    ax.set_title(f'Fraud Rate by {factorDict[factor]}', fontsize=20)
    ax.set_ylabel('Percentage')
    ax.set_xlabel(factorDict[factor])
    ax.legend(['No', 'Yes'], title='Is Fraud')
    ax.tick_params(axis='x', rotation=45)

#outputting the created plots
plt.tight_layout()
plt.show()

#making a violin plot
violinFactors = []
#looping through the potential causes, and grouping them
for factor in fraudFactors:
    dataGrouping = (
        data
        .groupby(factor)['Is_Fraud']
        .mean()
        .reset_index(name='Fraud_Rate')
        .assign(Factor=factor)
        .rename(columns={factor:'Category'})
    )
    violinFactors.append(dataGrouping)
#making a new DF
fraud_rates_df = pd.concat(violinFactors, ignore_index=True)


#using px.violin to make hovering yield more information
violin = px.violin(
    fraud_rates_df,
    x="Factor",
    y="Fraud_Rate",
    box=True,              
    points="all",          
    hover_data=["Category"]
)

#adding the meanlines. Hopefully this aids in readability/comprehension
violin.update_traces(meanline_visible=True)
#showing the plot
violin.show()



