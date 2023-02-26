import tweepy
import re
import time

MAX_RESULTS = 100 # API Max by default
TWEET_FIELDS = 'id,author_id,conversation_id,referenced_tweets,created_at'

def basic_size_filter(t):
    """removes hashtags and compares size
    """
    txt = t.text
    if not txt: return False
    txt = re.sub("#[A-Za-z0-9_]+","", txt).strip(" ").strip("\n")
    return len(txt) > 40

def run_query(client,
              query,
              params={},
              search_all=True,
              max_steps=10000):
    """
    Runs a query to get all results
    Args:
        client: tweepy Client
        conversation_id: string of twitter conv id
        search_params: params to update default
        search_all: bool recent vs all twitter api
        get_root: bool of whether to return root node as well
    Returns:
        unstructured tweets
        
    """
    search_params = dict(
        query=query,
        tweet_fields=TWEET_FIELDS,
        max_results=MAX_RESULTS
    )
    search_params.update(params)
    twitter_call = client.search_all_tweets if search_all else client.search_recent_tweets
    data = []
    
    cnt = 0
    for response in tweepy.Paginator(twitter_call, **search_params):
        cnt += 1
        if response.data: data.extend(response.data)
        if cnt >= max_steps: break
        if response.meta.get('next_token'): time.sleep(1)
    
    return data

def get_tweets(client, query, filter_fn=basic_size_filter):
    def filter_tweets(tweets):
        return [t for t in tweets if filter_fn(t)]

    params = dict()
    results = run_query(client,
        query=query,
        params=params,
        search_all=True,
        max_steps=1
    )
    results = filter_tweets(results)
    return results

def phrase_check(word):
    if " " in word:
        return f"\"{word}\""
    else: 
        return word
    
def bigram_query(keywords, maxlen=1024):
    base_piece = 'lang:en -is:retweet -has:media -is:reply '
    query_pieces = list()
    length_flag = True
    ongoing_len = len(base_piece) + 2
    for i in range(len(keywords)):
        for j in range(i):
            piece = f"({phrase_check(keywords[i])} {phrase_check(keywords[j])})"
            ongoing_len += len(piece) + 4
            length_flag = ongoing_len <= maxlen
            if not length_flag: break
            query_pieces.append(piece)
            
        if not length_flag: break
    return base_piece + "(" + " OR ".join(query_pieces) + ")"