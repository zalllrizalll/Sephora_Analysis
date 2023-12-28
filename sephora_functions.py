# Import beberapa library yang dibutuhkan
from turtle import st
import numpy as np # Library python yang umum digunakan untuk operasi numerik, terutama operasi array numerik multidimensi
import pandas as pd # Library python yang digunakan untuk manipulasi dan analisis data, terutama dengan struktur data seperti DataFrames.
from matplotlib import pyplot as plt # Library python untuk membuat visualisasi grafis. Modul 'pyplot' -> antarmuka membuat grafik dan plot
import seaborn as sns # Library python untuk membuat plot statistik yang indah dan informatif
from matplotlib.ticker import NullFormatter
import matplotlib as mpl
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel,BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from transformers import  AdamW, BertConfig
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup
import time
import datetime
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@st.cache()
def load_dataset():
    df_product_info = pd.read_csv("/Assets/product_info.csv")
    df_reviews_1 = pd.read_csv("/Assets/reviews_0_250.csv",index_col = 0, dtype={'author_id':'str'})
    df_reviews_2 = pd.read_csv("/Assets/reviews_250_500.csv",index_col = 0, dtype={'author_id':'str'})
    df_reviews_3 = pd.read_csv("/Assets/reviews_500_750.csv",index_col = 0, dtype={'author_id':'str'})
    df_reviews_4 = pd.read_csv("/Assets/reviews_750_1000.csv",index_col = 0, dtype={'author_id':'str'})
    df_reviews_5 = pd.read_csv("/Assets/reviews_1000_1500.csv",index_col = 0, dtype={'author_id':'str'})
    df_reviews_6 = pd.read_csv("/Assets/reviews_1500_end.csv",index_col = 0, dtype={'author_id':'str'})

    # Merge df reviews
    df_reviews = pd.concat([df_reviews_1,df_reviews_2,df_reviews_3,df_reviews_4,df_reviews_5,df_reviews_6],axis=0)

    # Lets check df_product_info which columns that similar with df_reviews
    cols_to_use = df_product_info.columns.difference(df_reviews.columns) # Identifikasi column-column yang terdapat di df_product_info tetapi tidak ada di df_reviews
    cols_to_use = list(cols_to_use) # Mengubah objek index ke dalam bentuk list
    cols_to_use.append('product_id') # Menambahkan column product_id pada cols_to_use

    # Menggabungkan df_reviews dan df_product_info[cols_to_use] berdasarkan kolom 'product_id'
    df = pd.merge(df_reviews, df_product_info[cols_to_use], how='outer', on=['product_id', 'product_id'])