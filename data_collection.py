import tweepy
import requests
import pandas as pd
import numpy as np
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Twitter/X API setup
def setup_twitter_api():
    auth = tweepy.OAuthHandler(os.getenv('TWITTER_CONSUMER_KEY'), os.getenv('TWITTER_CONSUMER_SECRET'))
    auth.set_access_token(os.getenv('TWITTER_ACCESS_TOKEN'), os.getenv('TWITTER_ACCESS_TOKEN_SECRET'))
    return tweepy.API(auth)

# Fetch social media sentiment
def fetch_social_media_data(location, query="local events"):
    api = setup_twitter_api()
    try:
        tweets = api.search_tweets(q=query, geocode=location, count=50, tweet_mode='extended')
        sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
        sentiments = []
        for tweet in tweets:
            try:
                text = tweet.full_text
                sentiment = sentiment_analyzer(text)[0]
                sentiments.append(sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score'])
            except AttributeError:
                continue
        return np.mean(sentiments) if sentiments else 0.0
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return 0.0

# Fetch weather data
def fetch_weather_data(lat, lon):
    api_key = os.getenv('OPENWEATHER_API_KEY')
    url = f'https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts&appid={api_key}&units=metric'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()['daily'][0]['temp']['day']  # Daily temperature
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return 20.0  # Default temperature

# Generate synthetic historical sales (placeholder)
def generate_historical_sales():
    return np.random.randint(500, 1500)  # Simulated sales data

# Integrate data
def integrate_data(city, location):
    # Parse location (e.g., "19.0760,72.8777,10mi" -> lat, lon)
    lat, lon = location.split(',')[0], location.split(',')[1]
    df = pd.DataFrame({
        'city': [city],
        'location': [location],
        'sentiment': [fetch_social_media_data(location)],
        'weather': [fetch_weather_data(lat, lon)],
        'historical_sales': [generate_historical_sales()]
    })
    return df

# Main execution (for testing)
if __name__ == "__main__":
    city = "Mumbai"
    location = "19.0760,72.8777,10mi"  # Mumbai coordinates
    df = integrate_data(city, location)
    print(df)