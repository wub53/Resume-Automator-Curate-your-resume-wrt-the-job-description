# NOTE: The utility files such as this one should be in the same directory as "scrapy.cfg" file !!!!

from scipy.spatial.distance import cosine
from transformers import RobertaTokenizer, RobertaModel
from chunkipy import TextChunker, TokenEstimator

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#model = RobertaModel.from_pretrained('roberta-base')

class RobertaTokenEstimator(TokenEstimator):
    def __init__(self):
        self.robert_tokenizer = tokenizer

    def estimate_tokens(self, text):
        return len(self.robert_tokenizer.encode(text))


