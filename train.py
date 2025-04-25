import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from evaluate import evaluate


# {'text': "My favourite food is anything I didn't have to cook myself.",
# 'labels': [27], 'id': 'eebbqej'}


# BCEWithLogitsLoss includes a sigmoid
def train_epoch(model: nn.Module, data_loader, optimizer=None, loss_fn=nn.BCEWithLogitsLoss()):
    model.train()
    if optimizer is None:
        optimizer = Adam(model.parameters())
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        logits = model(input_ids)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def train(model: nn.Module, train_data_loader, eval_data_loader, epochs, optimizer=None, loss_fn=nn.BCEWithLogitsLoss()):
    for i in range(epochs):
        total_loss = train_epoch(model, train_data_loader, optimizer, loss_fn)
        print(f"[Epoch {i+1}/{epochs}] Average loss: {total_loss}")
        report = evaluate(model, eval_data_loader, loss_fn)
        print(report)
    return model