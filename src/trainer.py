from typing import Optional, Tuple

import numpy as np
import transformers


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

    def _compute_metrics(self, p: transformers.EvalPrediction):
        head_count, head_total = 0, 0
        relation_count, relation_total = 0, 0
        for predictions, labels in zip(p.predictions, p.label_ids):
            head_logits, relation_logits = predictions[:2]
            heads, relations = labels[:2]

            count, total = compute_accuracy(head_logits, heads, ignore_index=-100)
            head_count += count
            head_total += total

            indices = np.where(heads == -100, 0, heads).reshape(*heads.shape, 1, 1)
            count, total = compute_accuracy(
                np.take_along_axis(relation_logits, indices, axis=2).squeeze(2),
                relations,
                ignore_index=-100,
            )
            relation_count += count
            relation_total += total

        head_accuracy = head_count / head_total
        relation_accuracy = relation_count / relation_total

        return {"head_accuracy": head_accuracy, "relation_accuracy": relation_accuracy}


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
