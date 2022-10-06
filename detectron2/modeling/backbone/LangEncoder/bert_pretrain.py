import torch
from torch import nn
from transformers import BertConfig, RobertaConfig, RobertaModel, BertModel

from .registry import register_lang_encoder


class BertEncoder(nn.Module):
    def __init__(self, cfg):
        super(BertEncoder, self).__init__()
        self.cfg = cfg
        self.bert_name = cfg['PRETRAINED']
        self.use_checkpoint = self.cfg.get('ENABLE_CHECKPOINT', False)

        if 'bert' in self.bert_name:
            config = BertConfig.from_pretrained(self.bert_name)
            config.gradient_checkpointing = self.use_checkpoint
            self.model = BertModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
        elif "roberta" in self.bert_name:
            config = RobertaConfig.from_pretrained(self.bert_name)
            config.gradient_checkpointing = self.use_checkpoint
            self.model = RobertaModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
        else:
            raise NotImplementedError

        self.language_dim = cfg['LANG_DIM']
        self.num_layers = cfg.get('N_LAYERS', 1)

    def forward(self, input_ids, attention_mask):
        input = input_ids
        mask = attention_mask

        # with padding, always 256
        outputs = self.model(
            input_ids=input,
            attention_mask=mask,
            output_hidden_states=True,
        )
        # outputs has 13 layers, 1 input layer and 12 hidden layers
        encoded_layers = outputs.hidden_states[1:]
        features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)

        # language embedding has shape [len(phrase), seq_len, language_dim]
        features = features / self.num_layers

        embedded = features * mask.unsqueeze(-1).float()
        aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

        ret = {
            "aggregate": aggregate,
            "embedded": embedded,
            "masks": mask,
            "last_hidden_state": encoded_layers[-1]
        }
        return ret


@register_lang_encoder
def build_bert_backbone(config_encoder, tokenizer, verbose, **kwargs):
    return BertEncoder(config_encoder)