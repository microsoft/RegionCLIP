import torch
from typing import Union, List
from transformers import AutoTokenizer

class BertTokenizer(object):
    def __init__(self, tokenizer_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77):

        if isinstance(texts, str):
            texts = [texts]

        results = []
        for text in texts:
            indexed_tokens = self.tokenizer.encode(text)
            if len(indexed_tokens) > context_length:
                tokens = indexed_tokens[:context_length-1] + [indexed_tokens[-1]]
            else:
                tokens = indexed_tokens + [0] * (context_length - len(indexed_tokens))
            result = torch.tensor(tokens).long()
            results.append(result)        
        return torch.stack(results, 0)

    def __call__(self, texts: Union[str, List[str]], context_length: int = 77):
        return self.tokenize(texts, context_length)
        
if __name__ == "__main__":
    tokenizer = BertTokenizer()
    tokens = tokenizer.tokenize(
        texts="The American alligator have a long, rounded snout that has upward facing nostrils at the end; this allows breathing to occur while the rest of the body is underwater. The young have bright yellow stripes on the tail; adults have dark stripes on the tail. It's easy to distinguish an alligator from a crocodile by the teeth.")
