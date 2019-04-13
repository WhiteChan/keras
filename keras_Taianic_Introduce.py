import urllib.request
import os
import numpy as np 
import pandas as pd 

url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
filepath = "data/titanic3.xls"
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url, filepath)
    print('downloaded: ', result)

all_df = pd.read_excel(filepath)
print(all_df[:2])