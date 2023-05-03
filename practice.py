#1 Pandas
import pandas as pd
import torch 

device = torch.device("cuda:0")   # 使用第一个 GPU 设备
if torch.cuda.is_available():
    device = torch.device("cuda")          # 如果有可用的 GPU 设备就使用 GPU
else:
    device = torch.device("cpu")           # 否则使用 CPU

path = "D:\python_project\\transformer\graduation_project_il\csv file\short_right_hand.csv"


file = pd.read_csv(path)
head = file.head()#查看前n行数据
print(head)
print(file.shape)#不包含第一行，第一行为索引
print(file.columns)#打印列的索引，即第一行
print(file.index)#RangeIndex(start=0, stop=5, step=1)，5代表行数（减去第一行）
print(file.dtypes)#每一列的数据类型

data1 = file.loc[0,"tactile64"]  ##row索引从0 开始，  列索引 直接指定名称
print("data1 = ",data1)


def cal(a,b):
    return a/b
path1 = "D:\python_project\\transformer\graduation_project_il\\template.csv"
file1 = pd.read_csv(path1)
head1 = file1.head()#查看前n行数据
print(head1)
print(file1.shape)#不包含第一行，第一行为索引
print(file1.columns)#打印列的索引，即第一行
print(file1.index)#RangeIndex(start=0, stop=5, step=1)，5代表行数（减去第一行）
print(file1.dtypes)#每一列的数据类型
data2 = file1[['age']].apply(cal,b=100)
file1['Prim']=0
print(file1)
data2 = file1[['age','gender']]
print(data2)
columns1 = torch.tensor(file1[['age','height']].values)##pandas 数据转化为torch张量
print(columns1.shape)
# columns, labels= selectColumnLabels(self.coordinates)

input_size = columns1.size()[0]
print('input_size = ' ,input_size)

maxsize = len(columns1)
print('maxsize = ' ,maxsize)


device = torch.device("cuda:0")   # 使用第一个 GPU 设备
if torch.cuda.is_available():
    device = torch.device("cuda")          # 如果有可用的 GPU 设备就使用 GPU
else:
    device = torch.device("cpu")           # 否则使用 CPU
print(device)
print(torch.version.cuda)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device("cuda")
    print("Device:", torch.cuda.get_device_name(device))
else:
    print("CUDA is not available")
    
print(torch.cuda.memory_allocated(device=device))
print(torch.cuda.max_memory_allocated(device=device))