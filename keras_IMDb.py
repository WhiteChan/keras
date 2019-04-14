import urllib.request
import os
import tarfile 

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

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

print(train_text[0])
print(y_train[0])

print(train_text[12501])
print(y_train[12501])

# 建立token
token = Tokenizer(num_words=2000)
token.fit_on_sequences(train_text)

print(token.document_count)
print(token.word_index)

# 使用token将文字转换成数字列表
