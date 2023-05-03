#1 Pandas
import pandas as pd
path = "D:\python_project\\transformer\graduation_project_il\csv file\\roll+1_2023-02-14-14-36-01-left-raw-joint_states.csv"
path1 = "D:\python_project\\transformer\graduation_project_il\csv file\\roll+1_2023-02-14-14-36-01-left-TactileGlove.csv"

file = pd.read_csv(path)
head = file.head()#查看前n行数据
print(head)
print(file.shape)#不包含第一行，第一行为索引
print(file.columns)#打印列的索引，即第一行
print(file.index)#RangeIndex(start=0, stop=5, step=1)，5代表行数（减去第一行）
print(file.dtypes)#每一列的数据类型

file1 = pd.read_csv(path1)
head1 = file1.head()#查看前n行数据
print(head)
print(file1.shape)#不包含第一行，第一行为索引
print(file1.columns)#打印列的索引，即第一行
print(file1.index)#RangeIndex(start=0, stop=5, step=1)，5代表行数（减去第一行）
print(file1.dtypes)#每一列的数据类型

# 删除多列
file1 = file1.drop(['time', '.header.seq', '.header.stamp.secs', '.header.stamp.nsecs',
       '.header.frame_id'], axis=1)

# 合并两个DataFrame
merged_df = pd.concat([file, file1], axis=1)

# 将结果保存到新的CSV文件中
merged_df.to_csv('D:\python_project\\transformer\graduation_project_il\csv file\short_right_hand.csv', index=False)