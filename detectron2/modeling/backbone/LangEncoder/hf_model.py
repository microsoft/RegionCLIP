import logging

from transformers import AutoConfig
from transformers import AutoModel

from .registry import register_lang_encoder

logger = logging.getLogger(__name__)


@register_lang_encoder
def lang_encoder(config_encoder, tokenizer, verbose, **kwargs):

    hf_model = None
    if config_encoder['LOAD_PRETRAINED']:
        hf_model = AutoModel.from_pretrained(config_encoder['HF_MODEL'])
    else:
        hf_config = AutoConfig.from_pretrained(config_encoder['HF_MODEL'])

        if 'CONFIG_OVERRIDE' in config_encoder:
            logger.warning(f'Override config: {config_encoder["CONFIG_OVERRIDE"]}')
            hf_config.update(config_encoder['CONFIG_OVERRIDE'])

        logger.info(f'HF model config: {hf_config}')
        hf_model = AutoModel.from_config(hf_config)

    return hf_model
