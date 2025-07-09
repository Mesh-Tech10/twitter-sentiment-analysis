# Twitter Sentiment Analysis Project
# A beginner-friendly implementation

import tweepy
import pandas as pd
import numpy as np
import re
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
import time
import json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TwitterSentimentAnalyzer:
    """
    A comprehensive Twitter sentiment analysis tool.
    This class handles data collection, processing, and analysis.
    """
    
    def __init__(self):
        """Initialize the analyzer with necessary components."""
        print("🚀 Initializing Twitter Sentiment Analyzer...")
        
        # Initialize sentiment analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Twitter API credentials (you'll need to set these)
        self.api = None
        
        # Data storage
        self.tweets_data = []
        
        print("✅ Analyzer initialized successfully!")
    
    def setup_twitter_api(self, consumer_key, consumer_secret, access_token, access_token_secret):
        """
        Set up Twitter API connection.
        You need to get these from Twitter Developer Portal.
        """
        try:
            # Authenticate with Twitter
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)
            
            # Create API object
            self.api = tweepy.API(auth, wait_on_rate_limit=True)
            
            # Test the connection
            self.api.verify_credentials()
            print("✅ Twitter API connected successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error connecting to Twitter API: {e}")
            return False
    
    def clean_text(self, text):
        """
        Clean and preprocess tweet text.
        Removes URLs, mentions, hashtags, and special characters.
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^A-Za-z\s]', '', text)
        
        # Convert to lowercase and remove extra spaces
        text = text.lower().strip()
        text = ' '.join(text.split())
        
        return text
    
    def analyze_sentiment_textblob(self, text):
        """
        Analyze sentiment using TextBlob.
        Returns polarity (-1 to 1) and subjectivity (0 to 1).
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment = 'Positive'
        elif polarity < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity
        }
    
    def analyze_sentiment_vader(self, text):
        """
        Analyze sentiment using VADER.
        Better for social media text with slang and emoticons.
        """
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Classify based on compound score
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'Positive'
        elif compound <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        return {
            'sentiment': sentiment,
            'compound': compound,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    
    def collect_tweets(self, query, count=100):
        """
        Collect tweets based on search query.
        """
        if not self.api:
            print("❌ Twitter API not set up. Please run setup_twitter_api() first.")
            return False
        
        print(f"🔍 Searching for tweets with query: '{query}'")
        tweets = []
        
        try:
            # Search for tweets
            for tweet in tweepy.Cursor(
                self.api.search_tweets,
                q=query,
                lang='en',
                result_type='recent',
                tweet_mode='extended'
            ).items(count):
                
                tweet_data = {
                    'id': tweet.id,
                    'created_at': tweet.created_at,
                    'text': tweet.full_text,
                    'user': tweet.user.screen_name,
                    'followers_count': tweet.user.followers_count,
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count,
                    'location': tweet.user.location
                }
                
                tweets.append(tweet_data)
            
            self.tweets_data = tweets
            print(f"✅ Collected {len(tweets)} tweets!")
            return True
            
        except Exception as e:
            print(f"❌ Error collecting tweets: {e}")
            return False
    
    def analyze_tweets(self):
        """
        Perform sentiment analysis on collected tweets.
        """
        if not self.tweets_data:
            print("❌ No tweets to analyze. Please collect tweets first.")
            return None
        
        print("🧠 Analyzing sentiment of tweets...")
        analyzed_tweets = []
        
        for tweet in self.tweets_data:
            # Clean the text
            cleaned_text = self.clean_text(tweet['text'])
            
            # Skip empty tweets
            if not cleaned_text:
                continue
            
            # Analyze with both methods
            textblob_result = self.analyze_sentiment_textblob(cleaned_text)
            vader_result = self.analyze_sentiment_vader(cleaned_text)
            
            # Combine results
            tweet['cleaned_text'] = cleaned_text
            tweet['textblob_sentiment'] = textblob_result['sentiment']
            tweet['textblob_polarity'] = textblob_result['polarity']
            tweet['textblob_subjectivity'] = textblob_result['subjectivity']
            tweet['vader_sentiment'] = vader_result['sentiment']
            tweet['vader_compound'] = vader_result['compound']
            tweet['vader_positive'] = vader_result['positive']
            tweet['vader_negative'] = vader_result['negative']
            tweet['vader_neutral'] = vader_result['neutral']
            
            analyzed_tweets.append(tweet)
        
        self.tweets_data = analyzed_tweets
        print(f"✅ Analyzed {len(analyzed_tweets)} tweets!")
        return analyzed_tweets
    
    def generate_report(self):
        """
        Generate a comprehensive sentiment analysis report.
        """
        if not self.tweets_data:
            print("❌ No analyzed data available.")
            return None
        
        df = pd.DataFrame(self.tweets_data)
        
        print("\n" + "="*50)
        print("📊 SENTIMENT ANALYSIS REPORT")
        print("="*50)
        
        # Basic statistics
        total_tweets = len(df)
        print(f"\n📈 Total Tweets Analyzed: {total_tweets}")
        
        # VADER sentiment distribution
        vader_counts = df['vader_sentiment'].value_counts()
        print(f"\n🎭 VADER Sentiment Distribution:")
        for sentiment, count in vader_counts.items():
            percentage = (count / total_tweets) * 100
            print(f"   {sentiment}: {count} ({percentage:.1f}%)")
        
        # TextBlob sentiment distribution
        textblob_counts = df['textblob_sentiment'].value_counts()
        print(f"\n🎭 TextBlob Sentiment Distribution:")
        for sentiment, count in textblob_counts.items():
            percentage = (count / total_tweets) * 100
            print(f"   {sentiment}: {count} ({percentage:.1f}%)")
        
        # Average scores
        avg_polarity = df['textblob_polarity'].mean()
        avg_subjectivity = df['textblob_subjectivity'].mean()
        avg_compound = df['vader_compound'].mean()
        
        print(f"\n📊 Average Scores:")
        print(f"   TextBlob Polarity: {avg_polarity:.3f} (-1=negative, +1=positive)")
        print(f"   TextBlob Subjectivity: {avg_subjectivity:.3f} (0=objective, 1=subjective)")
        print(f"   VADER Compound: {avg_compound:.3f} (-1=negative, +1=positive)")
        
        # Most engaging tweets
        top_tweets = df.nlargest(3, 'retweet_count')[['text', 'vader_sentiment', 'retweet_count']]
        print(f"\n🔥 Most Retweeted Tweets:")
        for i, (_, tweet) in enumerate(top_tweets.iterrows(), 1):
            print(f"   {i}. [{tweet['vader_sentiment']}] {tweet['text'][:100]}... ({tweet['retweet_count']} RTs)")
        
        return df
    
    def create_visualizations(self):
        """
        Create visualizations of the sentiment analysis results.
        """
        if not self.tweets_data:
            print("❌ No data to visualize.")
            return
        
        df = pd.DataFrame(self.tweets_data)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Twitter Sentiment Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. VADER Sentiment Distribution (Pie Chart)
        vader_counts = df['vader_sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # Green, Red, Gray
        axes[0, 0].pie(vader_counts.values, labels=vader_counts.index, autopct='%1.1f%%', 
                       colors=colors, startangle=90)
        axes[0, 0].set_title('VADER Sentiment Distribution')
        
        # 2. Sentiment over Time
        df['created_at'] = pd.to_datetime(df['created_at'])
        df_hourly = df.groupby([df['created_at'].dt.hour, 'vader_sentiment']).size().unstack(fill_value=0)
        df_hourly.plot(kind='bar', ax=axes[0, 1], color=['#2ecc71', '#e74c3c', '#95a5a6'])
        axes[0, 1].set_title('Sentiment Distribution by Hour')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Number of Tweets')
        axes[0, 1].legend(title='Sentiment')
        
        # 3. Polarity vs Subjectivity Scatter Plot
        scatter = axes[1, 0].scatter(df['textblob_polarity'], df['textblob_subjectivity'], 
                                   c=df['vader_compound'], cmap='RdYlGn', alpha=0.6)
        axes[1, 0].set_xlabel('Polarity (Negative ← → Positive)')
        axes[1, 0].set_ylabel('Subjectivity (Objective ← → Subjective)')
        axes[1, 0].set_title('Sentiment Polarity vs Subjectivity')
        plt.colorbar(scatter, ax=axes[1, 0], label='VADER Compound Score')
        
        # 4. Engagement vs Sentiment
        sentiment_engagement = df.groupby('vader_sentiment')[['retweet_count', 'favorite_count']].mean()
        sentiment_engagement.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Average Engagement by Sentiment')
        axes[1, 1].set_ylabel('Average Count')
        axes[1, 1].legend(['Retweets', 'Likes'])
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        print("📈 Visualizations created successfully!")
    
    def save_results(self, filename=None):
        """
        Save analysis results to CSV file.
        """
        if not self.tweets_data:
            print("❌ No data to save.")
            return
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_{timestamp}.csv"
        
        df = pd.DataFrame(self.tweets_data)
        df.to_csv(filename, index=False)
        print(f"💾 Results saved to {filename}")
    
    def load_sample_data(self):
        """
        Load sample data for testing (when Twitter API is not available).
        """
        sample_tweets = [
            "I love this new technology! It's amazing and revolutionary! 😍",
            "This is the worst product I've ever used. Completely disappointed 😞",
            "It's okay, nothing special but does the job.",
            "Absolutely fantastic! Best purchase I've made this year! 🎉",
            "Terrible customer service. Very frustrated with this company 😡",
            "Not bad, could be better though. Average experience.",
            "Mind-blowing innovation! This will change everything! 🚀",
            "Waste of money. Wouldn't recommend to anyone.",
            "Pretty good overall. Has some nice features.",
            "Outstanding quality and excellent design! Highly recommended! ⭐⭐⭐⭐⭐"
        ]
        
        # Create fake tweet data
        for i, text in enumerate(sample_tweets):
            tweet_data = {
                'id': i + 1,
                'created_at': datetime.now(),
                'text': text,
                'user': f'user_{i+1}',
                'followers_count': np.random.randint(100, 10000),
                'retweet_count': np.random.randint(0, 100),
                'favorite_count': np.random.randint(0, 200),
                'location': 'Sample Location'
            }
            self.tweets_data.append(tweet_data)
        
        print("✅ Sample data loaded successfully!")
        return True


# Main execution function
def main():
    """
    Main function to run the sentiment analysis.
    """
    print("🐦 Welcome to Twitter Sentiment Analysis!")
    print("This tool will help you analyze public opinion on any topic.\n")
    
    # Initialize analyzer
    analyzer = TwitterSentimentAnalyzer()
    
    # Ask user what they want to do
    print("Choose an option:")
    print("1. Use sample data (recommended for beginners)")
    print("2. Use real Twitter data (requires API credentials)")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "2":
        # Real Twitter data option
        print("\n🔑 Setting up Twitter API...")
        print("You'll need API credentials from developer.twitter.com")
        
        api_key = input("Enter your Twitter API Key: ").strip()
        api_secret = input("Enter your Twitter API Secret: ").strip()
        access_token = input("Enter your Access Token: ").strip()
        access_token_secret = input("Enter your Access Token Secret: ").strip()
        
        if api_key and api_secret and access_token and access_token_secret:
            # Try to connect to Twitter API
            if analyzer.setup_twitter_api(api_key, api_secret, access_token, access_token_secret):
                search_query = input("\nEnter search term (e.g., 'Python programming'): ").strip()
                if search_query:
                    tweet_count = input("How many tweets to analyze? (default: 50): ").strip()
                    tweet_count = int(tweet_count) if tweet_count.isdigit() else 50
                    
                    if analyzer.collect_tweets(search_query, count=tweet_count):
                        print("✅ Real Twitter data collected successfully!")
                    else:
                        print("❌ Failed to collect tweets. Using sample data instead.")
                        analyzer.load_sample_data()
                else:
                    print("❌ No search term provided. Using sample data instead.")
                    analyzer.load_sample_data()
            else:
                print("❌ Failed to connect to Twitter API. Using sample data instead.")
                analyzer.load_sample_data()
        else:
            print("❌ Missing API credentials. Using sample data instead.")
            analyzer.load_sample_data()
    else:
        # Sample data option (default)
        print("📝 Using sample data for demonstration...")
        analyzer.load_sample_data()
    
    # Analyze the tweets
    print("\n🧠 Starting sentiment analysis...")
    analyzer.analyze_tweets()
    
    # Generate report
    print("\n📊 Generating comprehensive report...")
    results_df = analyzer.generate_report()
    
    # Create visualizations
    print("\n📈 Creating beautiful visualizations...")
    analyzer.create_visualizations()
    
    # Save results
    print("\n💾 Saving results to file...")
    analyzer.save_results()
    
    print("\n🎉 Analysis complete! Check the generated files and visualizations.")
    print("\n💡 Next steps:")
    print("   • Look at the generated CSV file with your results")
    print("   • Check out the beautiful charts that were created")
    print("   • Try different search terms or topics")
    print("   • Experiment with the code to add new features!")
    
    if choice == "1":
        print("\n🚀 Ready for real data?")
        print("   1. Get free Twitter API access at developer.twitter.com")
        print("   2. Run this program again and choose option 2")
        print("   3. Analyze any topic you're interested in!")


if __name__ == "__main__":
    main()