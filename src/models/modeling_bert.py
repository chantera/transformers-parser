from transformers.models.bert import BertModel, BertPreTrainedModel

from modeling_utils import PreTrainedModelForParsing


class BertForParsing(BertPreTrainedModel, PreTrainedModelForParsing):
    def _init_encoder(self, config):
        self.bert = BertModel(config, add_pooling_layer=False)
