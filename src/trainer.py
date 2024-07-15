from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import transformers

from chuliu_edmonds import chuliu_edmonds_one_root

_DEFAULT_MAX_WORD_LENGTH = 128


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    max_word_length: int = _DEFAULT_MAX_WORD_LENGTH


class Trainer(transformers.Trainer):
    DEFAULT_LABEL_NAMES = ["heads", "relations"]

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("compute_metrics", self._compute_metrics)
        super().__init__(*args, **kwargs)

        if (
            self.args.eval_strategy != transformers.IntervalStrategy.NO
            or self.args.do_eval
            or self.args.do_predict
        ) and self.args.eval_do_concat_batches:
            raise ValueError("`eval_do_concat_batches` is not supported.")

        if not self.label_names:
            self.label_names = self.DEFAULT_LABEL_NAMES

        self._max_word_length = getattr(self.args, "max_word_length", _DEFAULT_MAX_WORD_LENGTH)

    def compute_loss(self, model, inputs, return_outputs=False):
        ret = super().compute_loss(model, inputs, return_outputs)

        if not return_outputs:
            return ret

        loss, outputs = ret

        def _pad(x):
            return pad(x, self._max_word_length, -float("inf"), dim=2)

        if isinstance(outputs, dict):
            outputs["head_logits"] = _pad(outputs["head_logits"])
            outputs["relation_logits"] = _pad(outputs["relation_logits"])
        else:
            outputs = outputs[:1] + (_pad(outputs[1]), _pad(outputs[2])) + outputs[3:]

        return loss, outputs

    def _compute_metrics(self, p: transformers.EvalPrediction):
        head_count, relation_count, uas_count, las_count, total = 0, 0, 0, 0, 0

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

        return {
            "head_accuracy": head_count / total,
            "relation_accuracy": relation_count / total,
            "UAS": uas_count / total,
            "LAS": las_count / total,
        }


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
    mask = np.zeros_like(head_logits)
    for i, length in enumerate(lengths):
        mask[i, :length, :length] = 1
    mask[:, np.arange(mask.shape[1]), np.arange(mask.shape[1])] = 0

    def softmax(x, axis=-1):
        y = np.exp(x - x.max(axis=axis, keepdims=True))
        y /= y.sum(axis=axis, keepdims=True)
        return y

    head_probs = softmax(np.where(mask, head_logits, -np.inf))
    heads = np.full((len(lengths), max(lengths)), -100)
    for i, length in enumerate(lengths):
        heads[i, :length] = chuliu_edmonds_one_root(head_probs[i, :length, :length])
    heads[:, 0] = -100

    indices = np.where(heads == -100, 0, heads)[..., None, None]
    relations = np.take_along_axis(relation_logits, indices, axis=2).squeeze(2).argmax(-1)
    for i, length in enumerate(lengths):
        relations[i, length:] = -100
    relations[:, 0] = -100

    return heads, relations
