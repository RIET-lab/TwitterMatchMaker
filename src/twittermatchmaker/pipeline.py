from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import pandas as pd
from .utils.keywords import KeywordExtractor
from .utils.twitter import bigram_query, get_tweets
from tqdm import tqdm
import time
import logging
import math

logger = logging.getLogger("pipeline")

class PipelineConfig:
    keyword_extractor = "rake"
    max_keywords = 3
    model_string = "sentence-transformers/sentence-t5-large"
    model_weights = "./model.pt"
    score_threshold = .8
    overlap_threshold = .98
    save_path = "./dataset.csv"
    tweepy_client = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wait_per_query = 60
    
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

        if self.tweepy_client is None:
            raise ValueError("Needs a tweepy_client to operate")
    
class Pipeline:
    def __init__(self, config):
        self.config = config
        self.extractor = KeywordExtractor.auto(config.keyword_extractor)
        self.model = SentenceTransformer(config.model_string)
        self.model.to(config.device)
        self.model.load_state_dict(torch.load(config.model_weights))
        self.tweepy_client = config.tweepy_client
        
        # Hack for using multiple credentials to decrease r.l. slowdown
        if not isinstance(self.tweepy_client, list):
            self.tweepy_client = [config.tweepy_client]
        print(f"Using {len(self.tweepy_client)} Tweepy Clients")
            
    def _get_query(self, claim):
        keywords = self.extractor.extract(claim)
        return bigram_query(keywords[:self.config.max_keywords])
    
    def _get_queries(self, claims):
        return map(self._get_query, claims)
    
    @staticmethod
    def create_record(tweet, claim, score):
        return dict(tweet_id=tweet.id, 
                    author_id=tweet.author_id, 
                    text=tweet.text, 
                    claim=claim,
                    score=score)
    
    def remove_noisy_duplicates(self, tweets, embs):
        scores = embs @ embs.T
        keep = [0]
        for i in range(1, len(embs)):
            if not len(set(np.where(scores[i, :i] >= self.config.overlap_threshold)[0])):
                keep.append(i)
        return [tweets[i] for i in keep], embs[keep, :]
        
    def _query_loop(self, query):
        tweets = None
        step_sleep = 5
        n_iter = math.ceil(self.config.wait_per_query / step_sleep)
        for i in range(n_iter):
            fails = 0
            for client in self.tweepy_client:
                try:
                    tweets = get_tweets(client, query)
                    return dict(status=200, tweets=tweets)
                except Exception as e:
                    try:
                        error_code = int(str(e)[:3])
                    except:
                        error_code = 400
                        
                    if error_code == 400:
                        return dict(status=error_code, tweets=None)
                    fails += 1
            if fails == len(self.tweepy_client) and i != n_iter-1:
                time.sleep(step_sleep)
        else:
            print("query stalled")
            return dict(status=error_code, tweets=None)
    
    def run(self, claims):
        queries = list(self._get_queries(claims))
        tweets = []
        print("Starting to extract tweets")
        for query in tqdm(queries):
            result = self._query_loop(query)
            if result["status"] == 200:
                tweets.extend(result["tweets"])
            elif result["status"] != 400:
                queries.append(query)
            else:
                continue
        tweets = list(set(tweets))
        print(f"Extracted {len(tweets)} tweets")
        
        tweet_embs = self.model.encode([t.text for t in tweets])
        tweets, tweet_embs = self.remove_noisy_duplicates(tweets, tweet_embs)
        print(f"After noisy duplicate removal. Extracted {len(tweets)} tweets")
        
        claim_embs = self.model.encode(claims)
        score_matrix = tweet_embs @ claim_embs.T
        matches = score_matrix >= self.config.score_threshold
        print(f"Computed all embeddings. There are a total of {matches.sum()} connections")
        
        records = []
        for i, claim in enumerate(claims):
            indeces = np.where(matches[:,i])[0]
            for idx in indeces:
                records.append(self.create_record(tweets[idx], claim, score_matrix[idx, i]))
        df = pd.DataFrame.from_records(records)
        df.to_csv(self.config.save_path, index=False)