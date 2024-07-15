import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.utils import ModelOutput


@dataclass
class ParsingModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    head_logits: torch.FloatTensor = None
    relation_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class PairwiseBilinear(nn.Module):
    """
    https://github.com/stanfordnlp/stanza/blob/v1.1.1/stanza/models/common/biaffine.py#L5
    https://pytorch.org/docs/2.3/_modules/torch/nn/modules/linear.html#Bilinear
    """

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in1_features, out_features, in2_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(0))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        d1, d2, out = self.in1_features, self.in2_features, self.out_features
        n1, n2 = input1.size(1), input2.size(1)
        # (b * n1, d1) @ (d1, out * d2) => (b * n1, out * d2)
        x1W = torch.mm(input1.view(-1, d1), self.weight.view(d1, out * d2))
        # (b, n1 * out, d2) @ (b, d2, n2) => (b, n1 * out, n2)
        x1Wx2 = x1W.view(-1, n1 * out, d2).bmm(input2.transpose(1, 2))
        y = x1Wx2.view(-1, n1, self.out_features, n2).transpose(2, 3)
        if self.bias is not None:
            y.add_(self.bias)
        return y  # (b, n1, n2, out)

    def extra_repr(self) -> str:
        return "in1_features={}, in2_features={}, out_features={}, bias={}".format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None
        )


class FeedForward(torch.nn.Module):
    def __init__(self, config, output_size=None):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size, output_size if output_size is not None else config.hidden_size
        )
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class BiaffineScorer(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()

        intermediate_size = getattr(config, "classifier_hidden_size", None)
        if intermediate_size is None:
            intermediate_size = config.hidden_size // 2

        dropout_prob = getattr(config, "classifier_dropout", None)
        if dropout_prob is None:
            dropout_prob = config.hidden_dropout_prob

        self.intermediate1 = FeedForward(config, intermediate_size)
        self.intermediate2 = FeedForward(config, intermediate_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.attention = PairwiseBilinear(intermediate_size, intermediate_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states1 = self.intermediate1(hidden_states)
        hidden_states1 = self.dropout(hidden_states1)
        hidden_states2 = self.intermediate2(hidden_states)
        hidden_states2 = self.dropout(hidden_states2)
        return self.attention(hidden_states1, hidden_states2)


class WordPooler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, word_offsets: torch.Tensor) -> torch.Tensor:
        batch_size, source_length, dim = hidden_states.size()
        target_length = word_offsets.max().item() + 1
        device = hidden_states.device

        def _take_all(offsets):
            for i, ofs in enumerate(offsets):
                if ofs == -100:
                    break
                yield (i, ofs)

        def _take_first(offsets):
            prev_ofs = -1
            for i, ofs in enumerate(offsets):
                if ofs == -100:
                    break
                if ofs != prev_ofs:
                    yield (i, ofs)
                    prev_ofs = ofs

        take = _take_first

        source_indices = []
        target_indices = []
        for i, offsets in enumerate(word_offsets):
            for token_idx, word_idx in take(offsets):
                source_indices.append(i * source_length + token_idx)
                target_indices.append(i * target_length + word_idx)

        output = torch.zeros((batch_size * target_length, dim), device=device)
        output = output.index_reduce(
            dim=0,
            index=torch.tensor(target_indices, device=device),
            source=hidden_states.view(batch_size * source_length, dim)[source_indices],
            reduce="mean",
            include_self=False,
        )
        output = output.view(batch_size, target_length, dim)

        return output


class BertForParsing(BertPreTrainedModel):
    def __init__(self, config, output_length: Optional[int] = None):
        super().__init__(config)
        self.output_length = output_length

        self.bert = BertModel(config, add_pooling_layer=False)
        self.pooler = WordPooler()
        self.head_scorer = BiaffineScorer(config, num_labels=1)
        self.relation_scorer = BiaffineScorer(config, num_labels=config.num_labels)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, PairwiseBilinear):
            module.reset_parameters()
        else:
            super()._init_weights(module)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        word_offsets: Optional[torch.Tensor] = None,
        heads: Optional[torch.Tensor] = None,
        relations: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], ParsingModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        assert word_offsets is not None
        hidden_states = self.pooler(outputs.last_hidden_state, word_offsets)

        word_lengths = word_offsets.max(dim=1)[0] + 1
        head_logits = self.head_scorer(hidden_states).squeeze(-1)
        head_logits = _mask_arc(head_logits, word_lengths)
        relation_logits = self.relation_scorer(hidden_states)

        max_seq_length = word_lengths.max().item()
        max_seq_length_in_batch = heads.size(1) if heads is not None else max_seq_length
        assert max_seq_length <= max_seq_length_in_batch

        loss = None
        if heads is not None and relations is not None:
            # Trim heads and relations exceeding max_seq_length (in the case of using `torch.nn.DataParallel`)  # noqa
            if max_seq_length < max_seq_length_in_batch:
                heads = heads[:, :max_seq_length].contiguous()
                relations = relations[:, :max_seq_length].contiguous()

            loss = _compute_loss(head_logits, relation_logits, heads, relations)

        output_length = max_seq_length_in_batch
        if self.output_length is not None:
            if self.output_length < output_length:
                raise ValueError(f"logits size exceeds output_length={self.output_length}")
            output_length = self.output_length

        if max_seq_length < output_length:
            head_logits = _expand_logits(head_logits, output_length)
            relation_logits = _expand_logits(relation_logits, output_length)

        if not return_dict:
            output = (head_logits, relation_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ParsingModelOutput(
            loss=loss,
            head_logits=head_logits,
            relation_logits=relation_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def _compute_loss(
    head_logits: torch.Tensor,
    relation_logits: torch.Tensor,
    heads: torch.Tensor,
    relations: torch.Tensor,
) -> torch.Tensor:
    loss_fct = nn.CrossEntropyLoss()
    head_loss = loss_fct(head_logits.view(-1, head_logits.size(-1)), heads.view(-1))

    indices = heads.masked_fill(heads == -100, 0)[..., None, None]
    relation_loss = loss_fct(
        torch.take_along_dim(relation_logits, indices, dim=2).view(-1, relation_logits.size(-1)),
        relations.view(-1),
    )

    return head_loss + relation_loss


def _mask_arc(
    logits: torch.Tensor, lengths: torch.Tensor, mask_diag: bool = False
) -> Optional[torch.Tensor]:
    with torch.no_grad():
        batch_size, max_length = lengths.numel(), lengths.max()
        mask = torch.zeros(batch_size, max_length, max_length)
        for i, length in enumerate(lengths):
            mask[i, :length, :length] = 1
        if mask_diag:
            mask.masked_fill_(torch.eye(max_length, dtype=torch.bool), 0)

    return logits.masked_fill(mask.logical_not().to(logits.device), -float("inf"))


def _expand_logits(logits: torch.Tensor, length: int) -> torch.Tensor:
    size = logits.size()
    new_size = (size[0], length, length) + size[3:]
    new_logits = torch.full(new_size, -float("inf"), device=logits.device)
    new_logits[:, : size[1], : size[2]] = logits
    return new_logits
