import itertools
from dataclasses import dataclass
from typing import Dict, Optional

import datasets

_UD_VERSION = "2.2"  # Used in CoNLL 2018 Shared Task
_BASE_URL = "https://raw.githubusercontent.com/UniversalDependencies"
_DATASETS = {
    "en_ewt": {
        "train": f"{_BASE_URL}/UD_English-EWT/r{_UD_VERSION}/en_ewt-ud-train.conllu",
        "dev": f"{_BASE_URL}/UD_English-EWT/r{_UD_VERSION}/en_ewt-ud-dev.conllu",
        "test": f"{_BASE_URL}/UD_English-EWT/r{_UD_VERSION}/en_ewt-ud-test.conllu",
    },
}

_SPLIT_MAPPING = {
    "train": datasets.Split.TRAIN,
    "dev": datasets.Split.VALIDATION,
    "test": datasets.Split.TEST,
}


def _resolve_splits(files):
    return {_SPLIT_MAPPING.get(k, k): v for k, v in files.items()}


@dataclass
class UDConfig(datasets.BuilderConfig):
    data_urls: Optional[Dict] = None
    encoding: str = "utf-8"
    encoding_errors: Optional[str] = None


class UD(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = UDConfig
    BUILDER_CONFIGS = [
        UDConfig(name=name, data_urls=_resolve_splits(urls)) for name, urls in _DATASETS.items()
    ]

    def _info(self):
        features = {
            "sentence_id": datasets.Value("string"),
            "text": datasets.Value("string"),
            # CoNLL-U format: https://universaldependencies.org/format.html
            "id": datasets.Sequence(datasets.Value("int32")),
            "form": datasets.Sequence(datasets.Value("string")),
            "lemma": datasets.Sequence(datasets.Value("string")),
            "upos": datasets.Sequence(datasets.Value("string")),
            "xpos": datasets.Sequence(datasets.Value("string")),
            "feats": datasets.Sequence(datasets.Value("string")),
            "head": datasets.Sequence(datasets.Value("int32")),
            "deprel": datasets.Sequence(
                datasets.features.ClassLabel(names=_UNIVERSAL_DEPENDENCIES_RELATIONS)
            ),
        }
        return datasets.DatasetInfo(features=datasets.Features(features))

    def _split_generators(self, dl_manager):
        # https://github.com/huggingface/datasets/blob/2.20.0/src/datasets/packaged_modules/text/text.py#L46
        data_urls = self.config.data_files or self.config.data_urls
        if not data_urls:
            raise ValueError(
                f"At least one data url must be specified, but got data_urls={data_urls}"
            )
        dl_manager.download_config.extract_on_the_fly = True
        data_files = dl_manager.download_and_extract(data_urls)
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            files = [dl_manager.iter_files(file) for file in files]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _generate_examples(self, files):
        idx = 0
        for file in itertools.chain.from_iterable(files):
            with open(
                file, encoding=self.config.encoding, errors=self.config.encoding_errors
            ) as f:
                for sent_id, text, tokens in _read_conll(f):
                    tokens = [token for token in tokens if token[0].isdecimal()]
                    example = {
                        "sentence_id": sent_id,
                        "text": text,
                        "id": [int(token[0]) for token in tokens],
                        "form": [token[1] for token in tokens],
                        "lemma": [token[2] for token in tokens],
                        "cpostag": [token[3] for token in tokens],
                        "postag": [token[4] for token in tokens],
                        "feats": [token[5] for token in tokens],
                        "head": [int(token[6]) for token in tokens],
                        "deprel": [token[7] for token in tokens],
                    }
                    yield (idx, example)
                    idx += 1


def _read_conll(lines):
    sent_id, text, tokens = None, None, []

    for line in lines:
        line = line.strip()
        if not line:
            if tokens:
                assert sent_id is not None and text is not None
                yield sent_id, text, tokens
                sent_id, text, tokens = None, None, []
        elif line.startswith("#"):
            if line.startswith("# sent_id = "):
                sent_id = line.split(" = ", maxsplit=1)[1]
            elif line.startswith("# text = "):
                text = line.split(" = ", maxsplit=1)[1]
        else:
            tokens.append(line.split("\t"))
    if len(tokens) > 1:
        assert sent_id is not None and text is not None
        yield sent_id, text, tokens


# https://universaldependencies.org/u/dep/
_UNIVERSAL_DEPENDENCIES_RELATIONS = [
    "acl",
    "acl:relcl",
    "advcl",
    "advcl:relcl",
    "advmod",
    "advmod:emph",
    "advmod:lmod",
    "amod",
    "appos",
    "aux",
    "aux:pass",
    "case",
    "cc",
    "cc:preconj",
    "ccomp",
    "clf",
    "compound",
    "compound:lvc",
    "compound:prt",
    "compound:redup",
    "compound:svc",
    "conj",
    "cop",
    "csubj",
    "csubj:outer",
    "csubj:pass",
    "dep",
    "det",
    "det:numgov",
    "det:nummod",
    "det:poss",
    "det:predet",
    "discourse",
    "dislocated",
    "expl",
    "expl:impers",
    "expl:pass",
    "expl:pv",
    "fixed",
    "flat",
    "flat:foreign",
    "flat:name",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nmod:npmod",
    "nmod:poss",
    "nmod:tmod",
    "nsubj",
    "nsubj:outer",
    "nsubj:pass",
    "nummod",
    "nummod:gov",
    "obj",
    "obl",
    "obl:agent",
    "obl:arg",
    "obl:lmod",
    "obl:npmod",
    "obl:tmod",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp",
]
