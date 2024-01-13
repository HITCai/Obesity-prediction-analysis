import pandas as pd

# 导入数据
data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# 查看缺失值
print(data.isnull().sum())

print('---------------------------')
# 查看重复值
print("是否有重复值：",any(data.duplicated()))

