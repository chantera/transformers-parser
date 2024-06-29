from contextlib import contextmanager
from itertools import chain
from typing import List, Optional, Tuple

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    PreTokenizedInput,
    PreTrainedTokenizerBase,
    TruncationStrategy,
)


def tokenize_pretokenized_input(
    text: str,
    words: List[str],
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[List[int], List[int]]:
    batch_input_ids, batch_offsets = batch_tokenize_pretokenized_input([text], [words], tokenizer)
    return batch_input_ids[0], batch_offsets[0]


def batch_tokenize_pretokenized_input(
    batch_text: List[str],
    batch_words: List[List[str]],
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[List[List[int]], List[List[int]]]:
    if getattr(tokenizer, "add_prefix_space", tokenizer.init_kwargs.get("add_prefix_space")):
        batch_text = [" " + text if not text[0].isspace() else text for text in batch_text]

    text_input = []
    try:
        for text, words in zip(batch_text, batch_words):
            ofs = 0
            for word in words:
                end = text.index(word, ofs) + len(word)
                text_input.append(text[ofs:end])
                ofs = end
            assert ofs == len(text)
    except ValueError as e:
        raise ValueError(f"Could not find all words in the text: {text=}, {words=}") from e

    with disable_add_prefix_space(tokenizer) as _tokenizer:
        input_ids = _tokenizer(text_input, add_special_tokens=False)["input_ids"]

    batch_input_ids = []
    batch_offsets = []
    ofs = 0
    for words in batch_words:
        chunk = input_ids[ofs : ofs + len(words)]
        batch_input_ids.append(list(chain.from_iterable(chunk)))
        batch_offsets.append(
            list(chain.from_iterable([i] * len(ids) for i, ids in enumerate(chunk)))
        )
        ofs += len(words)
    assert ofs == len(input_ids)

    return batch_input_ids, batch_offsets


@contextmanager
def disable_add_prefix_space(tokenizer):
    _tokenizer = tokenizer

    add_prefix_space = None
    if hasattr(tokenizer, "add_prefix_space"):
        add_prefix_space = tokenizer.add_prefix_space
        tokenizer.add_prefix_space = False

    if (add_prefix_space or tokenizer.init_kwargs.get("add_prefix_space")) or tokenizer.is_fast:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer.name_or_path, use_fast=True, add_prefix_space=False
        )

    yield tokenizer

    if add_prefix_space is not None:
        _tokenizer.add_prefix_space = add_prefix_space


def batch_prepare_for_model(
    batch_ids: List[PreTokenizedInput],
    tokenizer: PreTrainedTokenizerBase,
    add_special_tokens: bool = True,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    max_length: Optional[int] = None,
    stride: int = 0,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Optional[str] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_length: bool = False,
    verbose: bool = True,
) -> BatchEncoding:
    batch_outputs = {}  # type: ignore
    for ids in batch_ids:
        outputs = tokenizer.prepare_for_model(
            ids,
            add_special_tokens=add_special_tokens,
            padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=None,  # we pad in batch afterward
            return_attention_mask=False,  # we pad in batch afterward
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=None,  # We convert the whole batch to tensors at the end
            prepend_batch_axis=False,
            verbose=verbose,
        )

        for key, value in outputs.items():
            if key not in batch_outputs:
                batch_outputs[key] = []
            batch_outputs[key].append(value)

    batch_outputs = tokenizer.pad(
        batch_outputs,
        padding=padding_strategy.value,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        return_attention_mask=return_attention_mask,
    )

    batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

    return batch_outputs
