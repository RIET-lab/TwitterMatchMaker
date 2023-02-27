# TwitterMatchMaker
This package is designed to ease getting candidate datasets of tweets that match a collection of claims

## Install
First install pytorch. Use the instructions on their [site](https://pytorch.org/) to do so.

Next install the package via `pip install .`

## Running
You can see an example of how to use the pipeline in `example_script.py`
```
from twittermatchmaker.pipeline import PipelineConfig, Pipeline
import tweepy

twitter_credentials = dict() # Fill in with your own credentials and settings
tweepy_client = tweepy.Client(**twitter_credentials)

config = PipelineConfig(tweepy_client=tweepy_client, 
                        save_path="example.csv")
pipeline = Pipeline(config=config)

claims = ["Dogs are better than cats", "Pinnochio is a real boy"]
pipeline.run(claims)
```

You can configure the following flags in the config:

| Attribute | Inputs | Default | Description |
| --- | --- | --- | --- |
| keyword_extractor | {rake, keybert, yaneyuk} | rake | Which keyword extractor to use |
| max_keywords | int > 0 | 3 | extracts up to top n keywords per claim |
| model_string | string | sentence-transformers/sentence-t5-large | sentence transformer architecture for matching task |
| model_weights | string | ./model.pt | path to claim matching model's weights |
| score_threshold | 0 < float < 1 | .8 | threshold to capture candidate matches |
| save_path | string | ./dataset.csv | path to save resulting dataset |
| tweepy_client | tweept.Client | None | tweepy client to query twitter api |
| device | torch.device | torch.device("cuda:0" if torch.cuda.is_available() else "cpu") | which device to run claim matching model on |