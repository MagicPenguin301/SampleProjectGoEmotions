import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn=nn.BCEWithLogitsLoss()):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        total_loss = 0
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"]
            trues = batch["labels"]
            logits = model(input_ids)
            loss = loss_fn(logits, trues)
            total_loss += loss.item()
            probas = torch.sigmoid(logits)
            preds = (probas >= 0.5).float()
            true_labels.append(trues.cpu())
            pred_labels.append(preds.cpu())
        avg_loss = total_loss / len(data_loader)
        acc = accuracy_score(true_labels, pred_labels)
        prec, rec, f1, _ = precision_recall_fscore_support(
            torch.cat(true_labels).numpy(), torch.cat(pred_labels).numpy(), average="samples"
        )
    return {
        "average_loss": avg_loss, 
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1-score": f1,
    }
