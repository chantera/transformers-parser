import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from tokenization_utils import batch_prepare_for_model, batch_tokenize_pretokenized_input
from training_utils import LoggerCallback, setup_logger

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

    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    raw_dataset = load_dataset(args.dataset, detokenize=True, cache_dir=args.cache_dir)

    def preprocess(examples):
        batch_input_ids, batch_offsets = batch_tokenize_pretokenized_input(
            examples["text"], examples["form"], tokenizer
        )
        features = batch_prepare_for_model(batch_input_ids, tokenizer)
        features["word_offsets"] = batch_offsets
        return features

    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = raw_dataset.map(preprocess, batched=True, load_from_cache_file=False)

    model = AutoModel.from_pretrained(args.model, config=config)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset.get("train"),
        eval_dataset=dataset.get("validation"),
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


if __name__ == "__main__":
    CONFIG_FILE = Path(__file__).parents[1] / "training.conf"
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses(args_filename=CONFIG_FILE)
    main(args, training_args)
