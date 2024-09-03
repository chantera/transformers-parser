import importlib
from collections import OrderedDict

from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.models.auto.auto_factory import _LazyAutoMapping as _BaseLazyAutoMapping
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES


class _LazyAutoMapping(_BaseLazyAutoMapping):
    def _load_attr_from_module(self, model_type, attr):
        if "." in attr:
            module_name, attr = attr.rsplit(".", 1)
            if module_name not in self._modules:
                self._modules[module_name] = importlib.import_module(module_name)
            return getattr(self._modules[module_name], attr)

        return super()._load_attr_from_module(model_type, attr)


MODEL_FOR_PARSING_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "models.modeling_bert.BertForParsing"),
        ("roberta", "models.modeling_roberta.RobertaForParsing"),
    ]
)

MODEL_FOR_PARSING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_PARSING_MAPPING_NAMES)


class AutoModelForParsing(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PARSING_MAPPING
