import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

"""
sns 相关设置
@return:
"""
# 声明使用 Seaborn 样式
sns.set()
# 有五种seaborn的绘图风格，它们分别是：darkgrid, whitegrid, dark, white, ticks。默认的主题是darkgrid。
sns.set_style("whitegrid")
# 有四个预置的环境，按大小从小到大排列分别为：paper, notebook, talk, poster。其中，notebook是默认的。
sns.set_context('talk')
# 中文字体设置-黑体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 数据预处理，主要是针对缺失数据、类别数据的处理，方便直接进行建模
df_train = pd.read_csv('./贷款违约预测_train.csv')
df_testA = pd.read_csv('./贷款违约预测_testA.csv')

df_train['isDefault'].value_counts()

df_train.isnull().sum()
'''
数据存在不同程度的缺失 ，n系列数据缺失较为严重，此外，employmentLength、dti、pubRecBankruptcies、revolUtil存在小比例缺失，
postCode、employmentTitle、title 都只缺失一条
'''

# 查看特征的缺失程度
missing_series = df_train.isnull().sum()/df_train.shape[0]
missing_df = pd.DataFrame(missing_series).reset_index()
missing_df = missing_df.rename(columns={'index': 'col', 0: 'missing_pct'})
missing_df = missing_df.sort_values('missing_pct', ascending=False).reset_index(drop=True)

# 缺失率定义为0.8
threshold_features = 0.8
missing_col_num = missing_df[missing_df.missing_pct>=threshold_features].shape[0]
print('缺失率超过{}的变量个数为{}'.format(threshold_features, missing_col_num))

# 设置标题
plt.figure(figsize=(20, 10))
plt.title('缺失特征的分布图')
sns.barplot(data=missing_df[missing_df.missing_pct>0], x='col', y='missing_pct')
plt.ylabel('缺失率')
plt.show()

# 查看特征的数值特征有哪些，类别特征有哪些
numerical_fea = list(df_train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea, list(df_train.columns)))

print(numerical_fea)
print(category_fea)


# 这样的划分方式会将部分类别型特征识别为数值特征，例如：类别特征是数值的那种。这里我们使用一种方法：再次检测数值型特征中不同值个数，如果小于10，进行二次处理（看作类别特征）
# 建议采用如下划分方式：
# 划分数值型变量中的连续变量和分类变量
# 过滤数值型类别特征

def get_numerical_serial_fea(data, feas):
    numerical_serial_fea = []
    numerical_noserial_fea = []
    for fea in feas:
        temp = data[fea].nunique()
        if temp <= 10:
            numerical_noserial_fea.append(fea)
        else:
            numerical_serial_fea.append(fea)
    return numerical_serial_fea,numerical_noserial_fea

numerical_serial_fea, numerical_noserial_fea = get_numerical_serial_fea(df_train, numerical_fea)

print(numerical_serial_fea)
print(numerical_noserial_fea)

# 1.同值化数据处理
df_train.verificationStatus.value_counts() # 特征数据分布还算均匀

df_train.n11.value_counts() 
"""
0.0    729682
1.0       540
2.0        24
4.0         1
3.0         1
Name: n11, dtype: int64

分布相差悬殊，可以考虑分箱或者剔除该特征
"""
df_train.policyCode.value_counts()
"""
1.0    800000
Name: policyCode, dtype: int64

特征无用，全部是一个值
"""

# 查看特征中特征的单方差（同值化）性质
threshold_const = 0.95

const_list = [x for x in df_train.columns if x!='isDefault']
const_col = []
const_val = []

for col in const_list:
    # value_counts 的最多的一个样本类别的样本数
    max_samples_count = df_train[col].value_counts().iloc[0]
    # 总体非空样本数
    sum_samples_count = df_train[df_train[col].notnull()].shape[0]
    
    # 计算特征中类别最多的样本占比
    const_val.append(max_samples_count/sum_samples_count)
    # 过滤同值化特征
    if max_samples_count/sum_samples_count >= threshold_const:
        const_col.append(col)

const_val  = sorted(const_val)
const_val

print('常变量/同值化比例大于{}的特征个数为{}'.format(threshold_const, len(const_col)))
# 设置标题
plt.figure(figsize=(13, 5))
plt.title('同值化特征的分布图')
plt.plot(range(len(df_train.columns)-1), const_val)
plt.xlabel('特征个数')
plt.ylabel('同值化比例')
plt.show()

# 2.特征对应的整体分布情况
# 小于500个类别的特征进行整体分布的探索
for f in df_train.columns:
    if df_train[f].nunique()<500:
        print(f, '类型数：', df_train[f].nunique())
    
# 单特征探索
# 计算每个地区的违约率情况
df_bucket = df_train.groupby('regionCode')
bad_trend = pd.DataFrame()

bad_trend['total'] = df_bucket['isDefault'].count()
bad_trend['bad'] = df_bucket['isDefault'].sum()
bad_trend['bad_rate'] = round(bad_trend['bad']/bad_trend['total'], 4)*100
bad_trend = bad_trend.reset_index()
# 查看Top10的数据
bad_trend.sort_values(by='bad_rate', ascending=False).iloc[:10]


'''
可以看到存在部分地区的违约率高于平均值，
可以单独拿出Top地区做特征衍生
'''
# 计算subGrade的违约率情况
df_bucket = df_train.groupby('subGrade')
bad_trend = pd.DataFrame()

bad_trend['total'] = df_bucket['isDefault'].count()
bad_trend['bad'] = df_bucket['isDefault'].sum()
bad_trend['bad_rate'] = round(bad_trend['bad']/bad_trend['total'], 4)*100
bad_trend = bad_trend.reset_index()