import urllib.request
import os
import tarfile 

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten 
from keras.layers.embeddings import Embedding

# 下载IMDb数据
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath = "data/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url, filepath)
    print('downloaded: ', result)

# 解压文件
if not os.path.exists("data/aclImdb"):
    tfile = tarfile.open("data/aclImdb_v1.tar.gz", 'r:gz')
    result = tfile.extractall('data/')

# 用正则表达式删除文字中的HTML标签
import re
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)

# 读取IMDb文件
def read_files(filetype):
    path = "data/aclImdb/"
    file_list = []

    positive_path = path + filetype + "/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]
    
    negative_path = path + filetype + "/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]
    
    print('read ', filetype, ' files: ', len(file_list))

    all_labels = ([1] * 12500 + [0] * 12500)

    all_texts = []
    for fi in file_list:
        with open(fi, encoding = 'utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]

    return all_labels, all_texts

y_train, train_text = read_files("train")
y_test, test_text = read_files("test")

# print(train_text[0])
# print(y_train[0])

# print(train_text[12501])
# print(y_train[12501])

# 建立token
token = Tokenizer(num_words=2000)
token.fit_on_texts(train_text)

# print(token.document_count)
# print(token.index_word)

# 使用token将文字转换成数字列表
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)

# print(train_text[0])
# print(x_train_seq[0])

# 让转换后的数字长度相同
x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
x_test = sequence.pad_sequences(x_test_seq, maxlen=100)

# 多层感知器模型
model = Sequential()
model.add(Embedding(output_dim=32, input_dim=2000, input_length=100))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.35))

model.add(Dense(units=1, activation='sigmoid'))
print(model.summary())