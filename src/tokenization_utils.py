from contextlib import contextmanager
from itertools import chain
from typing import List, Tuple

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def tokenize_pretokenized_input(
    text: str,
    words: List[str],
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[List[int], List[int]]:
    return batch_tokenize_pretokenized_input([text], [words], tokenizer)[0]


def batch_tokenize_pretokenized_input(
    batch_text: List[str],
    batch_words: List[List[str]],
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[List[int], List[int]]]:
    if getattr(tokenizer, "add_prefix_space", tokenizer.init_kwargs.get("add_prefix_space")):
        batch_text = [" " + text if not text[0].isspace() else text for text in batch_text]

    text_input = []
    for text, words in zip(batch_text, batch_words):
        ofs = 0
        for word in words:
            end = text.index(word, ofs) + len(word)
            text_input.append(text[ofs:end])
            ofs = end
        assert ofs == len(text)

    with disable_add_prefix_space(tokenizer) as _tokenizer:
        input_ids = _tokenizer(text_input, add_special_tokens=False)["input_ids"]

    ret = []
    ofs = 0
    for words in batch_words:
        chunk = input_ids[ofs : ofs + len(words)]
        ids = list(chain.from_iterable(chunk))
        offsets = list(chain.from_iterable([i] * len(c) for i, c in enumerate(chunk)))
        ret.append((ids, offsets))
        ofs += len(words)
    assert ofs == len(input_ids)

    return ret


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
