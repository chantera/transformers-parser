import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    set_seed,
)
from transformers.utils import check_min_version

from models import BertForParsing
from tokenization_utils import batch_prepare_for_model, batch_tokenize_pretokenized_input
from trainer import Trainer, TrainingArguments
from training_utils import LoggerCallback, setup_logger

check_min_version("4.40.0")

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    dataset: str = "data/ptb_wsj"
    model: str = "bert-base-uncased"
    cache_dir: Optional[str] = None


def main(args: Arguments, training_args: TrainingArguments):
    setup_logger(training_args)
    logger.warning(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}")  # noqa  # fmt: skip
    logger.info(f"args: {args}")
    logger.info(f"training args: {training_args}")
    set_seed(training_args.seed)

    raw_dataset = load_dataset(args.dataset, detokenize=True, cache_dir=args.cache_dir)
    label_list = raw_dataset["train"].features["deprel"].feature.names

    config = AutoConfig.from_pretrained(args.model, num_labels=len(label_list))
    config.classifier_hidden_size = 512
    config.classifier_dropout = 0.5
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = raw_dataset.map(partial(preprocess, tokenizer=tokenizer), batched=True)

    # model = AutoModel.from_pretrained(args.model, config=config)
    model = BertForParsing.from_pretrained(args.model, config=config)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset.get("train"),
        eval_dataset=dataset.get("validation"),
        data_collator=DataCollator(tokenizer),
    )
    trainer.add_callback(LoggerCallback(logger))

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        result = trainer.train(resume_from_checkpoint=checkpoint)
        logger.info(f"train metrics: {result.metrics}")
        trainer.log_metrics("train", result.metrics)
        if training_args.save_strategy != "no":
            trainer.save_model()
            trainer.save_state()
            trainer.save_metrics("train", result.metrics)

    if training_args.do_eval:
        metrics = trainer.evaluate()
        logger.info(f"eval metrics: {metrics}")
        trainer.log_metrics("eval", metrics)
        if training_args.save_strategy != "no":
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        pass  # do nothing


def preprocess(examples, tokenizer):
    batch_input_ids, batch_offsets = batch_tokenize_pretokenized_input(
        examples["text"], examples["form"], tokenizer
    )
    features = batch_prepare_for_model(batch_input_ids, tokenizer)

    has_prefix = False
    first_token = features["input_ids"][0][0]
    if first_token == tokenizer.cls_token_id or first_token == tokenizer.bos_token_id:
        has_prefix = True
    else:
        raise ValueError("prefix token is needed to be used as a dummy root")

    has_suffix = False
    last_token = features["input_ids"][0][-1]
    if last_token == tokenizer.sep_token_id or last_token == tokenizer.eos_token_id:
        has_suffix = True

    for offsets in batch_offsets:
        if has_prefix:
            offsets[:] = [0] + [ofs + 1 for ofs in offsets]
        if has_suffix:
            offsets.append(-100)

    features["word_offsets"] = batch_offsets
    assert all(
        len(input_ids) == len(offsets)
        for input_ids, offsets in zip(features["input_ids"], features["word_offsets"])
    )

    features["heads"] = [[-100] + heads for heads in examples["head"]]
    features["relations"] = [[-100] + relations for relations in examples["deprel"]]

    return features


class DataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        features = [f.copy() for f in features]

        extra_fields = {}
        extra_field_names = {"word_offsets", "heads", "relations"}
        for k in list(features[0].keys()):
            if k in extra_field_names:
                extra_fields[k] = [f.pop(k) for f in features]

        batch = super().__call__(features)
        batch.update(extra_fields)

        def pad_to_max_length(xs, max_length):
            if self.tokenizer.padding_side == "right":
                xs = [x + [-100] * (max_length - len(x)) for x in xs]
            else:
                xs = [[-100] * (max_length - len(x)) + x for x in xs]
            return xs

        token_seq_length = len(batch["input_ids"][0])
        word_seq_length = max(max(offsets) for offsets in batch["word_offsets"]) + 1
        batch["word_offsets"] = pad_to_max_length(batch["word_offsets"], token_seq_length)
        batch["heads"] = pad_to_max_length(batch["heads"], word_seq_length)
        batch["relations"] = pad_to_max_length(batch["relations"], word_seq_length)

        return batch.convert_to_tensors(self.return_tensors)


if __name__ == "__main__":
    CONFIG_FILE = Path(__file__).parents[1] / "training.conf"
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses(args_filename=CONFIG_FILE)
    main(args, training_args)
