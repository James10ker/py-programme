import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

def load_data(file_path):
    x=pd.read_csv(file_path,sep=',')
    print("Data Loaded Successfully")

    return x

data=load_data('lib/data.csv')

print(data)

new_data=data.drop(columns=["customerID","gender","Dependents","Partner","PaperlessBilling","PaymentMethod"])
#删除 客户唯一编号,性别,是否有家属,是否有配偶,是否无纸化账单,支付方式
print(new_data)

print(new_data.columns)