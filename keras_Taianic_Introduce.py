import urllib.request
import os
import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout

url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
filepath = "data/titanic3.xls"
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url, filepath)
    # print('downloaded: ', result)

all_df = pd.read_excel(filepath)
# print(all_df[:2])

cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
all_df = all_df[cols]
# print(all_df[:2])

def PreprocessData(all_def):
    # 删除name字段
    df = all_df.drop(['name'], axis=1)

    # 找出null值的字段，将null数据替换成平均值
    # print(all_df.isnull().sum())

    age_mean = df['age'].mean()
    fare_mean = df['fare'].mean()
    df['age'] = df['age'].fillna(age_mean)
    df['fare'] = df['fare'].fillna(fare_mean)

    # 转换性别字段为0与1
    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)

    # 将embarked字段进行一位有效编码转换
    x_OneHot_df = pd.get_dummies(data=df, columns=['embarked'])

    # print(x_OneHot_df[:2])

    # DataFrame转换为Array
    ndarray = x_OneHot_df.values
    # print(ndarray.shape)

    # 提取freatures与label
    Label = ndarray[:, 0]
    Features = ndarray[:, 1:]

    # 将ndarray特征字段进行标准化

    # 标准化刻度
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

    # 标准化
    scaledFeatures = minmax_scale.fit_transform(Features)
    # print(scaledFeatures[:2])

    return scaledFeatures, Label

# 将数据以随机的方式分为训练数据与测试数据
msk = np.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]

# print('total: ', len(all_df), 'train: ', len(train_df), 'test: ', len(test_df))
train_Features, train_Label = PreprocessData(train_df)
test_Features, test_Label = PreprocessData(test_df)

# 建立模型
model = Sequential()

model.add(Dense(units=40, input_dim=9, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history = model.fit(x=train_Features, y=train_Label, validation_split=0.1, epochs=30, batch_size=30, verbose=2)

scores = model.evaluate(x=test_Features, y=test_Label)
print('acc = ', scores[1])

# 加入Jack和Rose的数据
Jack = pd.Series([0, 'Jack', 3, 'male', 23, 1, 0, 5.0000, 'S'])
Rose = pd.Series([1, 'Rose', 1, 'female', 20, 1, 0, 100.0000, 'S'])

JR_df = pd.DataFrame([list(Jack), list(Rose)], columns=['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'])

all_df = pd.concat([all_df, JR_df])

all_Features, Label = PreprocessData(all_df)
all_probability = model.predict(all_Features)

all_df_probability = all_df
all_df_probability.insert(len(all_df.columns), 'probability', all_probability)

print(all_df_probability[-2:])