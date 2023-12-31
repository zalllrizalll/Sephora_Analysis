# Import beberapa library yang dibutuhkan
import streamlit as st
import numpy as np # Library python yang umum digunakan untuk operasi numerik, terutama operasi array numerik multidimensi
import pandas as pd # Library python yang digunakan untuk manipulasi dan analisis data, terutama dengan struktur data seperti DataFrames.
from matplotlib import pyplot as plt # Library python untuk membuat visualisasi grafis. Modul 'pyplot' -> antarmuka membuat grafik dan plot
import seaborn as sns # Library python untuk membuat plot statistik yang indah dan informatif
from matplotlib.ticker import NullFormatter
import matplotlib as mpl
from sklearn.model_selection import train_test_split
import torch
# import torch.nn as nn
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
from wordcloud import WordCloud

@st.cache_data()
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

    # Rename the columns
    df.rename(columns={'review_text':'text', 'is_recommended':'label'},  inplace=True)

    # Get value text
    x = df.text.values

    # Get value label
    y = df.label.values

    return df, x, y

@st.cache_data()
def train_model(x, y):
    # Pertama, bagi data menjadi training (70%) dan sisa (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(x, y.astype(int), test_size=0.3, shuffle=True, random_state=42)

    # Kemudian, bagi data sisa tersebut menjadi validation dan testing (masing-masing 50% dari data sisa)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

    # Train to Model BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Data Training
    encoded_data_train = tokenizer.batch_encode_plus(
                            X_train,
                            add_special_tokens=True,
                            return_attention_mask=True,
                            padding=True,
                            max_length=128,
                            truncation=True,
                            return_tensors='pt'
                        )
    
    # Data Validation
    encoded_data_val = tokenizer.batch_encode_plus(
                            X_val,
                            add_special_tokens=True,
                            return_attention_mask=True,
                            padding=True,
                            max_length=128,
                            truncation=True,
                            return_tensors='pt'
                        )
    
    # Data Testing
    encoded_data_test = tokenizer.batch_encode_plus(
                            X_test,
                            add_special_tokens=True,
                            return_attention_mask=True,
                            padding=True,
                            max_length=128,
                            truncation=True,
                            return_tensors='pt'
                        )
    
    # Encoding Data Training
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(y_train)

    # Encoding Data Validation
    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(y_val)

    # Encoding Data Testing
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(y_test)

    # Dataset Training
    dataset_train = TensorDataset(input_ids_train, 
                                  attention_masks_train,
                                  labels_train)

    # Dataset Validation
    dataset_val = TensorDataset(input_ids_val, 
                                attention_masks_val,
                                labels_val)

    # Dataset Testing
    dataset_test = TensorDataset(input_ids_test, 
                                 attention_masks_test,
                                 labels_test)
    
    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", 
            num_labels = 2,   
            output_attentions = False, 
            output_hidden_states = False, )

    batch_size = 32
    # Train Dataloader
    train_dataloader = DataLoader(
        dataset_train,
        sampler=RandomSampler(dataset_train),
        batch_size=batch_size
    )

    # Validation Dataloader
    validation_dataloader = DataLoader(
        dataset_val,
        sampler=RandomSampler(dataset_val),
        batch_size=batch_size
    )

    # Test Dataloader
    test_dataloader = DataLoader(
        dataset_test,
        sampler=RandomSampler(dataset_test),
        batch_size=batch_size
    )

    # AdamW is an optimizer which is a Adam Optimzier with weight-decay-fix
    optimizer = AdamW(model.parameters(),
                    lr = 2e-5, 
                    eps = 1e-8 
                    )

    # Number of training epochs
    epochs = 10

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                num_warmup_steps = 0, # Default value in run_glue.py
                num_training_steps = total_steps)
    
    return model, scheduler

@st.cache_data()
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

@st.cache_data()
def analysis_review(sentence):
    if(sentence == 1):
        # Positive Review
        wordcloud = WordCloud(max_font_size = 160, margin=0, background_color = "white", colormap="Greens").generate(sentence)
        plt.figure(figsize=[10,10])
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.margins(x=0, y=0)
        plt.title("Positive Reviews Word Cloud")
        plt.show()
    else:
        # Negative Review
        wordcloud = WordCloud(max_font_size = 160, margin=0, background_color = "white", colormap="Reds").generate(sentence)
        plt.figure(figsize=[10,10])
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.margins(x=0, y=0)
        plt.title("Negative Reviews Word Cloud")
        plt.show()