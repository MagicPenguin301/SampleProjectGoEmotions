## A sample PyTorch/NLP project on the dataset [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions).

- This is a sample project to demonstrate basic coding skills in PyTorch and NLP.

- The model is a simple LSTM-based multi-label emotion classifier.

- The focus is on clean structure and readable code, rather than model performance.

### Running the Project
To train, evaluate, and test the model with default settings:

```bash
python main.py
```

To customize model hyperparameters:

```bash
python main.py --embedding_size 128 --hidden_size 256 --batch_size 32 --epochs 5 --shuffle
```
