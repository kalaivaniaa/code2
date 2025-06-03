from preprocess import load_data, prepare_dataloaders
from train import train_model
from evaluate import evaluate_model
from model import MHA_LSTM
import torch

X_train, X_test, y_train, y_test = load_data()
train_loader, test_loader = prepare_dataloaders(X_train, X_test, y_train, y_test)

model = MHA_LSTM(vocab_size=10000, embedding_dim=100, hidden_dim=64, n_heads=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_model(model, train_loader, device)
evaluate_model(model, test_loader, device)
