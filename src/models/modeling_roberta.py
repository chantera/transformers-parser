from transformers.models.roberta import RobertaModel, RobertaPreTrainedModel

from modeling_utils import PreTrainedModelForParsing


class RobertaForParsing(PreTrainedModelForParsing, RobertaPreTrainedModel):
    def _init_encoder(self, config):
        self.roberta = RobertaModel(config, add_pooling_layer=False)
