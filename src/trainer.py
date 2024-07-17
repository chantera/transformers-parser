from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import transformers
from transformers.training_args import ParallelMode

from chuliu_edmonds import chuliu_edmonds_one_root


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_logits_length: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.parallel_mode == ParallelMode.DISTRIBUTED and self.output_logits_length is None:
            raise ValueError("`output_logits_length` must be specified in distributed training")


class Trainer(transformers.Trainer):
    DEFAULT_LABEL_NAMES = ["heads", "relations"]

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("compute_metrics", self._compute_metrics)
        kwargs.setdefault("preprocess_logits_for_metrics", self._preprocess_logits_for_metrics)
        super().__init__(*args, **kwargs)

        if (
            self.args.eval_strategy != transformers.IntervalStrategy.NO
            or self.args.do_eval
            or self.args.do_predict
        ) and self.args.eval_do_concat_batches:
            raise ValueError("`eval_do_concat_batches` is not supported.")

        self._output_length = getattr(self.args, "output_logits_length", None)
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED and self._output_length is None:
            raise ValueError(
                "`args.output_logits_length` must be specified in distributed training"
            )

        if not self.label_names:
            self.label_names = self.DEFAULT_LABEL_NAMES

        self.last_prediction = None

    def _compute_metrics(self, p: transformers.EvalPrediction):
        head_count, relation_count, uas_count, las_count, total = 0, 0, 0, 0, 0
        outputs: List[Tuple[np.ndarray, np.ndarray]] = []

        for predictions, labels in zip(p.predictions, p.label_ids):
            head_logits, relation_logits = predictions[:2]
            heads, relations = labels[:2]

            head_count += compute_accuracy(head_logits, heads, ignore_index=-100)[0]
            indices = np.where(heads == -100, 0, heads)[..., None, None]
            relation_count += compute_accuracy(
                np.take_along_axis(relation_logits, indices, axis=2).squeeze(2),
                relations,
                ignore_index=-100,
            )[0]

            lengths = (heads[:, 1:] != -100).sum(axis=1) + 1
            pred_heads, pred_relations = parse(head_logits, relation_logits, lengths)
            correct_heads = np.logical_and(pred_heads == heads, heads != -100)
            correct_relations = np.logical_and(pred_relations == relations, relations != -100)
            uas_count += correct_heads.sum()
            las_count += np.logical_and(correct_heads, correct_relations).sum()
            total += (lengths - 1).sum()

            outputs.extend(
                (pred_heads[i, :n], pred_relations[i, :n]) for i, n in enumerate(lengths)
            )

        metrics = {
            "head_accuracy": head_count / total,
            "relation_accuracy": relation_count / total,
            "UAS": uas_count / total,
            "LAS": las_count / total,
        }
        self.last_prediction = outputs

        return metrics

    def _preprocess_logits_for_metrics(self, logits, labels):
        if self._output_length is None:
            return logits

        head_logits, relation_logits = logits[:2]
        head_logits = pad(head_logits, self._output_length, -float("inf"), dim=2)
        relation_logits = pad(relation_logits, self._output_length, -float("inf"), dim=2)
        return (head_logits, relation_logits) + logits[2:]


def pad(input: torch.Tensor, length: int, value: float, dim: int = -1) -> torch.Tensor:
    dim = dim if dim >= 0 else input.dim() + dim
    size = input.size(dim)
    if size > length:
        raise ValueError(f"input size ({size}) is greater than the specified length ({length})")

    p = (0, 0) * (input.dim() - dim - 1) + (0, length - size)
    return torch.nn.functional.pad(input, p, mode="constant", value=value)


def compute_accuracy(
    y: np.ndarray, t: np.ndarray, ignore_index: Optional[int] = None
) -> Tuple[int, int]:
    pred = np.argmax(y, axis=-1)
    if ignore_index is not None:
        mask = t == ignore_index
        ignore_cnt = np.sum(mask)
        pred[mask] = ignore_index
        count = np.sum(pred == t) - ignore_cnt
        total = t.size - ignore_cnt
    else:
        count = np.sum(pred == t)
        total = t.size
    return count, total


def parse(
    head_logits: np.ndarray,
    relation_logits: np.ndarray,
    lengths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    def softmax(x, axis=-1):
        y = np.exp(x - x.max(axis=axis, keepdims=True))
        y /= y.sum(axis=axis, keepdims=True)
        return y

    mask = np.zeros_like(head_logits)
    mask[:, np.arange(mask.shape[1]), np.arange(mask.shape[1])] = -np.inf
    head_probs = softmax(head_logits + mask)
    heads = np.full(head_probs.shape[:2], -100)
    for i, length in enumerate(lengths):
        heads[i, :length] = chuliu_edmonds_one_root(head_probs[i, :length, :length])
    heads[:, 0] = -100

    indices = np.where(heads == -100, 0, heads)[..., None, None]
    relations = np.take_along_axis(relation_logits, indices, axis=2).squeeze(2).argmax(-1)
    for i, length in enumerate(lengths):
        relations[i, length:] = -100
    relations[:, 0] = -100

    return heads, relations
