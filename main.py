from datasets import load_dataset
from utils import GoEmotionsDataset
from torch.utils.data.dataloader import DataLoader
from model import EmotionClassifierLSTM
from train import train
from evaluate import evaluate
import argparse

NUM_CLASSES = 27

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM on GoEmotions dataset")
    parser.add_argument("--embedding_size", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--hidden_size", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset during loading")
    return parser.parse_args()

def main():
    args = parse_args()

    full_data = load_dataset("google-research-datasets/go_emotions", "simplified")
    train_data = full_data["train"]
    eval_data = full_data["validation"]
    test_data = full_data["test"]

    train_dataset = GoEmotionsDataset(train_data, NUM_CLASSES)
    eval_dataset = GoEmotionsDataset(eval_data, NUM_CLASSES)
    test_dataset = GoEmotionsDataset(test_data, NUM_CLASSES)

    vocab_size = len(train_dataset.word2id)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = EmotionClassifierLSTM(vocab_size, args.embedding_size, args.hidden_size, args.num_classes)
    model = train(model, train_loader, eval_loader, args.epochs)

    test_report = evaluate(model, test_loader)
    print(test_report)

if __name__ == "__main__":
    main()