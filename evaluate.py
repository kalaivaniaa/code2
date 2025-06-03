from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

def evaluate_model(model, test_loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds += list((outputs.cpu().numpy() > 0.5).astype(int).flatten())
            labels += list(y_batch.numpy().flatten())

    print("Accuracy:", accuracy_score(labels, preds))
    print("Precision:", precision_score(labels, preds))
    print("Recall:", recall_score(labels, preds))
    print("F1 Score:", f1_score(labels, preds))
