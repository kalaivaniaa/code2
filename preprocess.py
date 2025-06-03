from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

def clean_text(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([w for w in text.split() if w.lower() not in stop_words])

def load_data():
    df = pd.read_csv("HASOC_Tamil_dataset.csv")
    df['text'] = df['text'].apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(df['text']).toarray()
    y = df['label'].apply(lambda x: 1 if x == 'OFF' else 0).values
    return train_test_split(X, y, test_size=0.1, random_state=42)

def prepare_dataloaders(X_train, X_test, y_train, y_test):
    X_train = torch.tensor(X_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    return DataLoader(train_data, batch_size=32, shuffle=True), DataLoader(test_data, batch_size=32)
