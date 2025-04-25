## Sample DL Project on the dataset [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions).

- This is just a sample project to demonstrate some basic skills in DL and NLP.

- The task and the model are simple and the performance is not highly relevant.

### Running
To train, evaluate and test the model on the dataset, simply run the following command:

```bash
python main.py
```

or with custom arguments:

```bash
python main.py --embedding_size 128 --hidden_size 256 --batch_size 32 --epochs 5 --shuffle
```