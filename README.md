# transformers-parser

A graph-based dependency parser with a Transformer-based encoder, implemented using [transformers](https://github.com/huggingface/transformers).

## Installation

```sh
$ git clone https://github.com/chantera/transformers-parser
$ cd transformers-parser
$ pip install -r requirements.txt
```

## Usage

Training and inference are performed using [src/train.py](src/train.py).

### Data Preparation

A dataset used for the training script must be a collection of entries, each of which consists of the following fields:

- `text` (string): *raw text*
- `form` (list of string): *words*
- `head` (list of int): *head indices*
- `deprel` (list of string): *dependency relations*

Below is an example represented in JSON:

```json
{
  "text": "Tokyo is the capital of Japan.",
  "form": ["Tokyo", "is", "the", "capital", "of", "Japan", "."],
  "head": [4, 4, 4, 0, 6, 4, 4],
  "deprel": ["nsubj", "cop", "det", "root", "case", "nmod", "punct"]
}
```

A dataset can be formatted in JSON Lines or prepared using a [dataset loading script](https://huggingface.co/docs/datasets/dataset_script), as provided in [data/ptb_wsj](data/ptb_wsj) and [data/ud](data/ud).

**Notes**:

- For JSON Lines datasets, `train.jsonl`, `validation.jsonl`, and `test.jsonl` files must be placed in the dataset directory.
- For PTB, `train.conll`, `validation.conll`, and `test.conll` files must be placed in [data/ptb_wsj](data/ptb_wsj).
- For UD, train/dev/test splits are automatically downloaded from [Universal Dependencies](https://github.com/UniversalDependencies).

### Training

The training script utilizes `transformers.Trainer` (See the official [documentation](https://huggingface.co/docs/transformers/main_classes/trainer) for details).
Other than `transformers.TrainingArguments`, you can specify `dataset` and `model`.
Below is an example of training a parser on the Penn Treebank:

```sh
$ torchrun --nproc_per_node 4 src/train.py \
    --dataset ./data/ptb_wsj \
    --model roberta-large \
    --output_dir ./output \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --max_grad_norm 0.5 \
    --eval_strategy epoch \
    --eval_steps 1 \
    --save_strategy epoch \
    --save_steps 1 \
    --save_total_limit 5 \
    --metric_for_best_model UAS \
    --load_best_model_at_end \
    --output_logits_length 128 \
    --do_train \
    --do_eval \
    --do_predict \
    --seed 42
```

**Notes**:

- The default values for `transformers.TrainingArguments` are redefined in [training.conf](training.conf).
- `output_logits_length` must be specified in distributed training to align logits in length.
- UAS/LAS scores evaluated through the training script are <u>not calculated using the CoNLL evaluation scripts</u>, and thus some special treatments, such as for punctuation, are not taken into account. See [Evaluation](#evaluation) for calculating official UAS/LAS scores.

### Evaluation

UAS/LAS scores using the CoNLL evaluation scripts can be evaluated using [eval/evaluate.py](eval/evaluate.py).
Below is an example of evaluating predictions on the Penn Treebank:

```sh
$ python eval/evaluate.py ./data/ptb_wsj/test.conll ./output/test_predictions.jsonl
```

### Performance

#### PTB

| Model         | UAS   | LAS   |
| ------------- | :---: | :---: |
| roberta-large | 97.30 | 95.75 |
