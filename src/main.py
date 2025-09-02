import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

pd.set_option('future.no_silent_downcasting', True)

def load_data(file_path):
    x=pd.read_csv(file_path,sep=',')
    print("Data Loaded Successfully")

    return x

data=load_data('lib/data.csv')

print(data)

new_data=data.drop(columns=["customerID","gender","Dependents","Partner","PaperlessBilling","PaymentMethod"])
#删除 客户唯一编号,性别,是否有家属,是否有配偶,是否无纸化账单,支付方式
print(new_data)
new_data.to_csv('lib/output.csv')


print(new_data.columns)

# 将性别列转换为数值型，Male为0，Female为1
new_data['PhoneService'] = new_data['PhoneService'].replace({'Yes': 1, 'No': 0})
new_data['MultipleLines'] = new_data['MultipleLines'].replace({'Yes': 1, 'No': 0,'No phone service':2})
new_data['InternetService'] = new_data['InternetService'].replace({ 'No': 0 ,'DSL': 1,'Fiber optic':2})
new_data['OnlineSecurity'] = new_data['OnlineSecurity'].replace({ 'No': 0 ,'Yes': 1,'No internet service':2})
new_data['OnlineBackup'] = new_data['OnlineBackup'].replace({ 'No': 0 ,'Yes': 1,'No internet service':2})
new_data['DeviceProtection'] = new_data['DeviceProtection'].replace({ 'No': 0 ,'Yes': 1,'No internet service':2})
new_data['TechSupport'] = new_data['TechSupport'].replace({ 'No': 0 ,'Yes': 1,'No internet service':2})
new_data['StreamingTV'] = new_data['StreamingTV'].replace({ 'No': 0 ,'Yes': 1,'No internet service':2})
new_data['StreamingMovies'] = new_data['StreamingMovies'].replace({ 'No': 0 ,'Yes': 1,'No internet service':2})
new_data['Contract'] = new_data['Contract'].replace({ 'Month-to-month': 0 ,'One year': 1,'Two year':2})
new_data['Churn'] = new_data['Churn'].replace({ 'No': 0 ,'Yes': 1})

new_data.to_csv('lib/output2.csv')






