import pandas as pd
df=pd.read_csv("employees.csv")
print(df)

import matplotlib.pyplot as plt

#boxplot for outliers in age

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.boxplot(df["Age"].dropna())
plt.title("Age")

#boxplot for outliers in salary
plt.subplot(1,2,2)
plt.boxplot(df["Salary"].dropna())
plt.title("Salary")

#plt.show()


#IQR METHOD
df_capped= df.copy()
for column in ['Age','Salary']:
    q1=df_capped[column].quantile(0.25)
    q3=df_capped[column].quantile(0.75)
    iqr=q3-q1
    lower_bound=q1-(1.5*iqr)
    upper_bound=q1+(1.5*iqr)
    df_capped[column]=df_capped[column].apply( 
        lambda x:upper_bound if x>upper_bound else (lower_bound if x<lower_bound else x)
    )

print('Data after capping using IQR')
print(df_capped)


#Median method
df_replaced= df.copy()
for column in ['Age','Salary']:
    q1=df_capped[column].quantile(0.25)
    q3=df_capped[column].quantile(0.75)
    iqr=q3-q1
    lower_bound=q1-(1.5*iqr)
    upper_bound=q1+(1.5*iqr)
    median = df_replaced[column].median()
    df_replaced[column]=df_replaced[column].apply( 
        lambda x:median if x>upper_bound else (median if x<lower_bound else x)
    )
print('Data after capping using median')
print(df_replaced)
