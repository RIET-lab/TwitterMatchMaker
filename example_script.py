from twittermatchmaker.pipeline import PipelineConfig, Pipeline
import tweepy

twitter_credentials = dict() # Fill in with your own credentials and settings
tweepy_client = tweepy.Client(**twitter_credentials)

config = PipelineConfig(tweepy_client=tweepy_client, 
                        save_path="example.csv")
pipeline = Pipeline(config=config)

claims = ["Dogs are better than cats", "Pinnochio is a real boy"]
pipeline.run(claims)