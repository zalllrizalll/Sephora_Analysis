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
from transformers.tokenization_utils_base import AddedToken
import time
import datetime
import io
import base64
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from wordcloud import WordCloud

@st.cache_data
def load_dataset():
    df_product_info = pd.read_csv("Assets/product_info.csv")
    df_reviews_1 = pd.read_csv("Assets/reviews_0_250.csv",index_col = 0, dtype={'author_id':'str'})
    df_reviews_2 = pd.read_csv("Assets/reviews_250_500.csv",index_col = 0, dtype={'author_id':'str'})
    df_reviews_3 = pd.read_csv("Assets/reviews_500_750.csv",index_col = 0, dtype={'author_id':'str'})
    df_reviews_4 = pd.read_csv("Assets/reviews_750_1000.csv",index_col = 0, dtype={'author_id':'str'})
    df_reviews_5 = pd.read_csv("Assets/reviews_1000_1500.csv",index_col = 0, dtype={'author_id':'str'})
    df_reviews_6 = pd.read_csv("Assets/reviews_1500_end.csv",index_col = 0, dtype={'author_id':'str'})

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

@st.cache_data
def hundformatter(x, pos):
    return str(round(x / 1e4, 1))

@st.cache_data
def create_feedback_plot_streamlit(df):
    # Convert 'submission_time' to datetime and extract year
    df['submission_time'] = pd.to_datetime(df['submission_time'])
    df['year'] = df['submission_time'].dt.year

    # Create a Streamlit app
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))

    total_feedback = df.groupby('year').sum(numeric_only=True)['total_feedback_count'].reset_index()

    sns.pointplot(data=total_feedback, x='year', y='total_feedback_count', color="blue", label="Total all feedback", ax=ax1)

    total_pos_feedback = df.groupby('year').sum(numeric_only=True)['total_pos_feedback_count'].reset_index()
    sns.pointplot(data=total_pos_feedback, x='year', y='total_pos_feedback_count', color="green", label="Total Positive Feedback", ax=ax1)

    total_neg_feedback = df.groupby('year').sum(numeric_only=True)['total_neg_feedback_count'].reset_index()
    sns.pointplot(data=total_neg_feedback, x='year', y='total_neg_feedback_count', color="red", label="Total Negative Feedback", ax=ax1)

    ax1.yaxis.set_major_formatter(hundformatter)
    ax1.set_ylabel("Total feedback in Thousands")
    ax1.set_xlabel("Years")
    ax1.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)

@st.cache_data
def plot_helpfulness_vs_recommendation(df):
    # Create a Streamlit figure
    fig, ax = plt.subplots()
    
    colors = {'0.0': 'red', '1.0': 'green'}
    
    # Use Seaborn's barplot within the figure
    sns.barplot(data=df, y='helpfulness', x='label', palette=colors, ax=ax)
    
    # Display the plot in Streamlit
    st.pyplot(fig)

@st.cache_data
def preprocessing_data(df):
    missing = []
    unique = []
    types = []
    variables = []
    count = []

    for item in df.columns:
        variables.append(item)
        missing.append(df[item].isnull().sum())
        unique.append(df[item].nunique())
        types.append(df[item].dtypes)
        count.append(len(df[item]))

    output = pd.DataFrame({
        'variable': variables,
        'dtype': types,
        'count': count,
        'unique': unique,
        'missing': missing,
    })

    df_info = output.sort_values("missing", ascending=False).reset_index(drop=True)

    # Specify columns to drop
    cols_to_drop = """variation_desc
    sale_price_usd
    value_price_usd
    child_max_price
    child_min_price
    review_title"""

    # Remove leading space in each column name
    cols_list = [col.strip() for col in cols_to_drop.split("\n")]

    # Drop the specified columns
    df.drop(columns=cols_list, axis=1, inplace=True)

    # Drop rows with missing values
    df.dropna(axis=0, inplace=True)

    return df

@st.cache_data
def train_model(x, y):
    # Check for NaN or inf values in labels
    if not pd.Series(y).isnull().all() and not np.isfinite(y).all():
        # Handle or remove NaN or inf values in labels
        y = pd.Series(y).dropna().astype(int).values

    # Trim x to match the length of y
    x = x[:len(y)]

    if len(x) != len(y):
        st.write(f"Length of x: {len(x)}, Length of y: {len(y)}")
        raise ValueError("Lengths of x and y must be the same.")

    # Create a DataFrame with 'text' and 'label' columns
    df = pd.DataFrame({'text': x, 'label': y})

    # Count the number of positive and negative instances in the original DataFrame
    pos_count = df['label'].sum()  # Assuming label 1 represents positive instances
    neg_count = len(df) - pos_count

    # Determine the minimum count for sampling
    min_count = min(pos_count, neg_count)

    # Sample 20,000 rows for both positive and negative labels
    pos_sampled = df[df['label'] == 1].sample(20000, replace=True)
    neg_sampled = df[df['label'] == 0].sample(20000, replace=True)

    # Concatenate pos and neg samples
    df_sampled = pd.concat([pos_sampled, neg_sampled], axis=0)

    # Extract the labels
    y = df_sampled['label'].astype(int).values

    if len(df_sampled) != len(y):
        raise ValueError("Lengths of x and y must be the same.")

    # Convert 'text' column to a list
    X = df_sampled['text'].tolist()

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, df_sampled['label'].astype(int), test_size=0.3, shuffle=True, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)
 
    # Convert X_train to strings
    x_train = [str(item) for item in X_train]

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
    
    return X_train, X_val, X_test, y_train, y_val, y_test

@st.cache(hash_funcs={AddedToken: lambda x: 0})
def train_and_predict_sentiment(review, x_train, y_train, tokenizer, model):
  # Check if y_train is not null and contains finite values
    if not pd.Series(y_train).isnull().all() and not np.isfinite(y_train).all():
        # Handle or remove NaN or inf values in labels
        y_train = pd.Series(y_train).dropna().astype(int).values

    # Trim x_train to match the length of y_train
    x_train = x_train[:len(y_train)]

    if len(x_train) != len(y_train):
        st.write(f"Length of x_train: {len(x_train)}, Length of y_train: {len(y_train)}")
        raise ValueError("Lengths of x_train and y_train must be the same.")

    # Create a DataFrame with 'text' and 'label' columns
    df_train = pd.DataFrame({'text': x_train, 'label': y_train})

    # Count the number of positive and negative instances in the original DataFrame
    pos_count = df_train['label'].sum()  # Assuming label 1 represents positive instances
    neg_count = len(df_train) - pos_count

    # Determine the minimum count for sampling
    min_count = min(pos_count, neg_count)

    # Sample 20,000 rows for both positive and negative labels
    pos_sampled = df_train[df_train['label'] == 1].sample(min_count, replace=True)
    neg_sampled = df_train[df_train['label'] == 0].sample(min_count, replace=True)

    # Convert the 'label' column to integers during sampling
    pos_sampled['label'] = pos_sampled['label'].astype(int)
    neg_sampled['label'] = neg_sampled['label'].astype(int)

    # Concatenate pos and neg samples
    df_sampled = pd.concat([pos_sampled, neg_sampled], axis=0)

    # Extract the labels
    y_train_sampled = df_sampled['label'].astype(int).values

    if len(df_sampled) != len(y_train_sampled):
        raise ValueError("Lengths of x_train and y_train_sampled must be the same.")

    # Convert 'text' column to a list
    x_train_sampled = df_sampled['text'].tolist()
    x_train_final = df_sampled['text'].tolist()

    # Split the data into training, validation, and test sets (adjust as needed)
    x_train_final, x_temp, y_train_final, y_temp = train_test_split(
        x_train_sampled, y_train_sampled, test_size=0.3, shuffle=True, random_state=42
    )

    # Encode training data
    encoded_data_train = tokenizer.batch_encode_plus(
        x_train_final,
        add_special_tokens=True,
        return_attention_mask=True,
        padding=True,
        max_length=128,
        truncation=True,
        return_tensors='pt'
    )

    # Extract input_ids, attention_masks
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']

    # Assuming you have labels in y_train_final, adjust this part based on your data
    labels = torch.tensor(y_train_final).long()

    # Training the model (replace this with your actual training code)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(3):  # Replace 3 with your desired number of epochs
        model.train()
        optimizer.zero_grad()
        logits = model(input_ids_train, attention_mask=attention_masks_train).logits
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

    # Encode test data
    encoded_data_test = tokenizer.batch_encode_plus(
        [review],
        add_special_tokens=True,
        return_attention_mask=True,
        padding=True,
        max_length=128,
        truncation=True,
        return_tensors='pt'
    )

    # Extract input_ids, attention_masks
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']

    # Make predictions
    with torch.no_grad():
        logits = model(input_ids_test, attention_mask=attention_masks_test).logits

    # Apply softmax to get probabilities
    probs = softmax(logits, dim=1)

    # Predicted labels
    predictions = torch.argmax(probs, dim=1).numpy()

    return predictions
@st.cache_data
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

@st.cache_data
def generate_wordcloud(text, sentiment_label):
    if sentiment_label == 1:
        # Positive Review
        color_map = "Greens"
    elif sentiment_label == 0:
        # Negative Review
        color_map = "Reds"
    else:
        st.error("Invalid sentiment label. Use 0 for negative or 1 for positive.")
        return None

    # Convert each item to a string, handling potential float values
    text = [str(item) for item in text]

    wordcloud = WordCloud(
        max_words=25,          # Adjust as needed
        max_font_size=80,       # Adjust as needed
        margin=0,
        background_color="darkgrey",
        colormap=color_map
    ).generate(' '.join(text))

    fig, ax = plt.subplots()
    plt.figure(figsize=[10, 10])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0)

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)

    # Encode the bytes as base64 to make it serializable
    img_str = base64.b64encode(img_bytes.read()).decode('utf-8')

    # Close the Matplotlib figure to free up resources
    plt.close()

    return img_str