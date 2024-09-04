import itertools
import os
import re
from dataclasses import dataclass
from typing import Optional

import datasets
from datasets.data_files import DataFilesDict, sanitize_patterns
from nltk.tokenize.treebank import TreebankWordDetokenizer


@dataclass
class PtbWsjConfig(datasets.BuilderConfig):
    detokenize: bool = True
    encoding: str = "utf-8"
    encoding_errors: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.data_files is None:
            self.data_files = DataFilesDict.from_local_or_remote(
                sanitize_patterns(
                    {
                        datasets.Split.TRAIN: "train.conll",
                        datasets.Split.VALIDATION: "validation.conll",
                        datasets.Split.TEST: "test.conll",
                    }
                ),
                base_path=os.path.abspath(
                    os.path.expanduser(self.data_dir) if self.data_dir else "data/ptb_wsj"
                ),
            )


class PtbWsj(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = PtbWsjConfig

    def _info(self):
        features = {
            "sentence_id": datasets.Value("string"),
            # CoNLL-X format: https://aclanthology.org/W06-2920/
            "id": datasets.Sequence(datasets.Value("int32")),
            "form": datasets.Sequence(datasets.Value("string")),
            "lemma": datasets.Sequence(datasets.Value("string")),
            "cpostag": datasets.Sequence(datasets.Value("string")),
            "postag": datasets.Sequence(datasets.Value("string")),
            "feats": datasets.Sequence(datasets.Value("string")),
            "head": datasets.Sequence(datasets.Value("int32")),
            "deprel": datasets.Sequence(
                datasets.features.ClassLabel(names=_STANFORD_TYPED_DEPENDENCIES)
            ),
        }
        if self.config.detokenize:
            features["text"] = datasets.Value("string")
        return datasets.DatasetInfo(features=datasets.Features(features))

    def _split_generators(self, dl_manager):
        # https://github.com/huggingface/datasets/blob/2.20.0/src/datasets/packaged_modules/text/text.py#L46
        if not self.config.data_files:
            raise ValueError(
                f"At least one data file must be specified, but got data_files={self.config.data_files}"  # noqa: E501
            )
        dl_manager.download_config.extract_on_the_fly = True
        data_files = dl_manager.download_and_extract(self.config.data_files)
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            files = [dl_manager.iter_files(file) for file in files]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _generate_examples(self, files):
        detokenizer = TreebankWordDetokenizer() if self.config.detokenize else None
        idx = 0
        for file in itertools.chain.from_iterable(files):
            with open(
                file, encoding=self.config.encoding, errors=self.config.encoding_errors
            ) as f:
                for i, tokens in enumerate(_read_conll(f)):
                    example = {
                        "sentence_id": f"{os.path.basename(file)}:{i}",
                        "id": [int(token[0]) for token in tokens],
                        "form": [token[1] for token in tokens],
                        "lemma": [token[2] for token in tokens],
                        "cpostag": [token[3] for token in tokens],
                        "postag": [token[4] for token in tokens],
                        "feats": [token[5] for token in tokens],
                        "head": [int(token[6]) for token in tokens],
                        "deprel": [token[7] for token in tokens],
                    }
                    if self.config.detokenize:
                        example["form"] = [token_to_text(t) for t in example["form"]]
                        example["text"] = detokenizer.detokenize(example["form"])
                    yield (idx, example)
                    idx += 1


def _read_conll(lines):
    tokens = []
    for line in lines:
        line = line.strip()
        if not line:
            if tokens:
                yield tokens
                tokens = []
        elif line.startswith("#"):
            continue
        else:
            tokens.append(line.split("\t"))
    if len(tokens) > 1:
        yield tokens


def token_to_text(token):
    text = _TOKEN_TO_TEXT_MAPPING.get(token)
    if text is not None:
        return text
    text = token
    for regex, repl in _TOKEN_TO_TEXT_SUBS:
        text = regex.sub(repl, text)
    return text


_TOKEN_TO_TEXT_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
}

_TOKEN_TO_TEXT_SUBS = [
    (re.compile(r"``"), '"'),
    (re.compile(r"''"), '"'),
]


# Stanford typed dependencies (ver 3.3.0)
# https://github.com/stanfordnlp/CoreNLP/blob/v3.3.0/doc/lexparser/StanfordDependenciesManual.tex
_STANFORD_TYPED_DEPENDENCIES = [
    "acomp",
    "advcl",
    "advmod",
    "agent",
    "amod",
    "appos",
    "arg",
    "aux",
    "auxpass",
    "cc",
    "ccomp",
    "comp",
    "conj",
    "cop",
    "csubj",
    "csubjpass",
    "dep",
    "det",
    "discourse",
    "dobj",
    "expl",
    "goeswith",
    "infmod",
    "iobj",
    "mark",
    "mod",
    "mwe",
    "neg",
    "nn",
    "npadvmod",
    "nsubj",
    "nsubjpass",
    "num",
    "number",
    "obj",
    "parataxis",
    "partmod",
    "pcomp",
    "pobj",
    "poss",
    "possessive",
    "preconj",
    "predet",
    "prep",
    "prepc",
    "prt",
    "punct",
    "quantmod",
    "rcmod",
    "ref",
    "root",
    "sdep",
    "subj",
    "tmod",
    "xcomp",
    "xsubj",
]
