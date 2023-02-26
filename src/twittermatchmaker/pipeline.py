from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import pandas as pd
from .utils.keywords import KeywordExtractor
from .utils.twitter import bigram_query, get_tweets
from tqdm import tqdm
import time
import logging

logger = logging.getLogger("pipeline")

class PipelineConfig:
    keyword_extractor = "rake"
    max_keywords = 3
    model_string = "sentence-transformers/sentence-t5-large"
    model_weights = "./model.pt"
    score_threshold = .8
    save_path = "./dataset.csv"
    tweepy_client = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
    
class Pipeline:
    def __init__(self, config=PipelineConfig()):
        self.config = config
        self.extractor = KeywordExtractor.auto(config.keyword_extractor)
        self.model = SentenceTransformer(config.model_string)
        self.model.to(config.device)
        self.model.load_state_dict(torch.load(config.model_weights))
        self.tweepy_client = config.tweepy_client
            
    def get_query(self, claim):
        keywords = self.extractor.extract(claim)
        return bigram_query(keywords[:self.config.max_keywords])
    
    def get_queries(self, claims):
        return map(self.get_query, claims)
    
    @staticmethod
    def create_record(tweet, claim):
        return dict(tweet_id=tweet.id, author_id=tweet.author_id, text=tweet.text, claim=claim)
    
    def run(self, claims):
        queries = self.get_queries(claims)
        tweets = []
        print("Starting to extract tweets")
        for query in tqdm(queries):
            for _ in range(12):
                try:
                    tweets.extend(get_tweets(self.tweepy_client, query))
                    break
                except:
                    time.sleep(5)
                    continue
            else:
                print("query failed")
        print(f"Extracted {len(tweets)} tweets")
        
        tweet_embs = self.model.encode([t.text for t in tweets])
        claim_embs = self.model.encode(claims)
        score_matrix = tweet_embs @ claim_embs.T
        matches = score_matrix >= self.config.score_threshold
        print(f"Computed all embeddings. There are a total of {matches.sum()} connections")
        
        records = []
        for i, claim in enumerate(claims):
            indeces = np.where(matches[:,i])[0]
            for idx in indeces:
                records.append(self.create_record(tweets[idx], claim))
        df = pd.DataFrame.from_records(records)
        df.to_csv(self.config.save_path, index=False)