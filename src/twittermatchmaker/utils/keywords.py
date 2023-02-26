import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from multi_rake import Rake
from keybert import KeyBERT

class KeywordExtractor:
    def extract(self, text):
        raise NotImplementedError()
    
    @staticmethod
    def auto(model, **kwargs):
        model = model.lower()
        if model == "rake":
            return RakeKeywordExtractor(**kwargs)
        elif model == "yaneyuk":
            return YaneyukKeywordExtractor(**kwargs)
        elif model == "keybert":
            return KeyBertKeywordExtractor(**kwargs)
        else:
            return ValueError(f"Model {model} has not been implemented")
        
class KeyBertKeywordExtractor(KeywordExtractor):
    def __init__(self):
        self.model = KeyBERT(model='all-mpnet-base-v2')
        
    def extract(self, text, **kwargs):
        params = dict(
            keyphrase_ngram_range=(1, 3), 
            stop_words='english',
            highlight=False,
            top_n=10
        )
        params.update(kwargs)
        keywords = self.model.extract_keywords(text, **params)
        keywords_list = list(dict(keywords).keys())
        return keywords_list

class RakeKeywordExtractor(KeywordExtractor):
    def __init__(self):
        self.rake = Rake()
    
    def extract(self, text):
        keywords = self.rake.apply(text)
        keywords = [kw for (kw, _) in keywords]
        return keywords

class YaneyukKeywordExtractor(KeywordExtractor):
    def __init__(self, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained("yanekyuk/bert-keyword-extractor")
        self.model = AutoModelForTokenClassification.from_pretrained("yanekyuk/bert-keyword-extractor")
        self.model.to(device)

    def extract(self, text, threshold=.3):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k,v in inputs.items()}
        out = self.model(**inputs)

        keywords = []
        start, end = None, None
        for i, vals in enumerate(torch.softmax(out.logits[0], dim=-1).detach().numpy()):
            if vals[1] >= threshold:
                start = i
                end = i
            elif vals[2] >= threshold:
                end = i
            elif start is not None:
                keywords.append(self.tokenizer.decode(token_ids=inputs["input_ids"][0][start:end+1]))
                start = None
                end = None
        return keywords