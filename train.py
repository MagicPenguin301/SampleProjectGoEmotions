from datasets import load_dataset
import torch, utils


train_data = load_dataset(
    "google-research-datasets/go_emotions", "simplified", split="train"
)
# {'text': "My favourite food is anything I didn't have to cook myself.", 'labels': [27], 'id': 'eebbqej'}


