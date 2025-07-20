"""
Social media channels module for gathering influencer chatter from Reddit and X.
"""
import os
import time
from typing import List, Dict, Optional
import praw
import tweepy
from datetime import datetime, timedelta

from .utils import (
    get_env_var, retry_on_failure, create_data_dir, save_jsonl, 
    load_jsonl, logger, sanitize_text
)

class RedditScraper:
    """Scrape Reddit for relevant discussions and influencer content."""
    
    def __init__(self):
        client_id = get_env_var("REDDIT_CLIENT_ID")
        client_secret = get_env_var("REDDIT_CLIENT_SECRET")
        user_agent = get_env_var("REDDIT_USER_AGENT", "ZeroClickCompass/1.0")
        
        if not all([client_id, client_secret]):
            logger.warning("Reddit API credentials not configured")
            self.reddit = None
        else:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def search_subreddits(self, query: str, subreddits: List[str] = None, 
                         limit: int = 100, time_filter: str = "month") -> List[Dict]:
        """Search for posts across multiple subreddits."""
        if not self.reddit:
            logger.warning("Reddit API not available")
            return []
        
        if subreddits is None:
            subreddits = [
                'marketing', 'SEO', 'contentmarketing', 'digitalmarketing',
                'entrepreneur', 'startups', 'business', 'technology'
            ]
        
        all_posts = []
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                search_results = subreddit.search(query, limit=limit, time_filter=time_filter)
                
                for post in search_results:
                    post_data = self._extract_post_data(post)
                    if post_data:
                        all_posts.append(post_data)
                
                logger.info(f"Found {len(all_posts)} posts in r/{subreddit_name}")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error scraping r/{subreddit_name}: {e}")
        
        return all_posts
    
    def _extract_post_data(self, post) -> Optional[Dict]:
        """Extract relevant data from a Reddit post."""
        try:
            return {
                'id': post.id,
                'title': sanitize_text(post.title),
                'content': sanitize_text(post.selftext),
                'author': str(post.author) if post.author else '[deleted]',
                'subreddit': str(post.subreddit),
                'score': post.score,
                'upvote_ratio': post.upvote_ratio,
                'num_comments': post.num_comments,
                'created_utc': post.created_utc,
                'url': f"https://reddit.com{post.permalink}",
                'platform': 'reddit',
                'engagement_score': self._calculate_engagement_score(post)
            }
        except Exception as e:
            logger.error(f"Error extracting post data: {e}")
            return None
    
    def _calculate_engagement_score(self, post) -> float:
        """Calculate engagement score for a post."""
        score = post.score
        comments = post.num_comments
        upvote_ratio = post.upvote_ratio
        
        # Simple engagement formula
        engagement = (score * upvote_ratio) + (comments * 2)
        return engagement
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def get_comments(self, post_id: str, limit: int = 50) -> List[Dict]:
        """Get comments for a specific post."""
        if not self.reddit:
            return []
        
        try:
            post = self.reddit.submission(id=post_id)
            post.comments.replace_more(limit=0)  # Remove MoreComments objects
            
            comments = []
            for comment in post.comments.list()[:limit]:
                comment_data = {
                    'id': comment.id,
                    'content': sanitize_text(comment.body),
                    'author': str(comment.author) if comment.author else '[deleted]',
                    'score': comment.score,
                    'created_utc': comment.created_utc,
                    'parent_id': comment.parent_id,
                    'platform': 'reddit'
                }
                comments.append(comment_data)
            
            return comments
            
        except Exception as e:
            logger.error(f"Error getting comments for post {post_id}: {e}")
            return []

class TwitterScraper:
    """Scrape X (Twitter) for relevant discussions and influencer content."""
    
    def __init__(self):
        bearer_token = get_env_var("TWITTER_BEARER_TOKEN")
        api_key = get_env_var("TWITTER_API_KEY")
        api_secret = get_env_var("TWITTER_API_SECRET")
        access_token = get_env_var("TWITTER_ACCESS_TOKEN")
        access_token_secret = get_env_var("TWITTER_ACCESS_TOKEN_SECRET")
        
        if not all([bearer_token, api_key, api_secret, access_token, access_token_secret]):
            logger.warning("Twitter API credentials not configured")
            self.client = None
        else:
            try:
                self.client = tweepy.Client(
                    bearer_token=bearer_token,
                    consumer_key=api_key,
                    consumer_secret=api_secret,
                    access_token=access_token,
                    access_token_secret=access_token_secret,
                    wait_on_rate_limit=True
                )
            except Exception as e:
                logger.error(f"Error initializing Twitter client: {e}")
                self.client = None
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def search_tweets(self, query: str, max_results: int = 100, 
                     start_time: datetime = None) -> List[Dict]:
        """Search for tweets matching the query."""
        if not self.client:
            logger.warning("Twitter API not available")
            return []
        
        if start_time is None:
            start_time = datetime.now() - timedelta(days=30)
        
        try:
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                start_time=start_time,
                tweet_fields=['created_at', 'public_metrics', 'author_id', 'lang'],
                user_fields=['username', 'name', 'verified', 'public_metrics'],
                expansions=['author_id']
            )
            
            if not tweets.data:
                return []
            
            # Create user lookup
            users = {user.id: user for user in tweets.includes['users']} if 'users' in tweets.includes else {}
            
            tweet_data = []
            for tweet in tweets.data:
                user = users.get(tweet.author_id)
                tweet_info = self._extract_tweet_data(tweet, user)
                if tweet_info:
                    tweet_data.append(tweet_info)
            
            logger.info(f"Found {len(tweet_data)} tweets for query: {query}")
            return tweet_data
            
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            return []
    
    def _extract_tweet_data(self, tweet, user) -> Optional[Dict]:
        """Extract relevant data from a tweet."""
        try:
            metrics = tweet.public_metrics
            
            return {
                'id': tweet.id,
                'content': sanitize_text(tweet.text),
                'author_id': tweet.author_id,
                'author_username': user.username if user else None,
                'author_name': user.name if user else None,
                'author_verified': user.verified if user else False,
                'author_followers': user.public_metrics['followers_count'] if user else 0,
                'created_at': tweet.created_at.isoformat(),
                'retweet_count': metrics['retweet_count'],
                'like_count': metrics['like_count'],
                'reply_count': metrics['reply_count'],
                'quote_count': metrics['quote_count'],
                'lang': tweet.lang,
                'url': f"https://twitter.com/user/status/{tweet.id}",
                'platform': 'twitter',
                'engagement_score': self._calculate_tweet_engagement(metrics)
            }
        except Exception as e:
            logger.error(f"Error extracting tweet data: {e}")
            return None
    
    def _calculate_tweet_engagement(self, metrics: Dict) -> float:
        """Calculate engagement score for a tweet."""
        retweets = metrics.get('retweet_count', 0)
        likes = metrics.get('like_count', 0)
        replies = metrics.get('reply_count', 0)
        quotes = metrics.get('quote_count', 0)
        
        # Weighted engagement formula
        engagement = (retweets * 2) + likes + (replies * 3) + (quotes * 2)
        return engagement
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def get_user_tweets(self, username: str, max_results: int = 100) -> List[Dict]:
        """Get recent tweets from a specific user."""
        if not self.client:
            return []
        
        try:
            user = self.client.get_user(username=username)
            if not user.data:
                return []
            
            tweets = self.client.get_users_tweets(
                id=user.data.id,
                max_results=max_results,
                tweet_fields=['created_at', 'public_metrics', 'lang'],
                exclude=['retweets', 'replies']
            )
            
            if not tweets.data:
                return []
            
            tweet_data = []
            for tweet in tweets.data:
                tweet_info = self._extract_tweet_data(tweet, user.data)
                if tweet_info:
                    tweet_data.append(tweet_info)
            
            return tweet_data
            
        except Exception as e:
            logger.error(f"Error getting tweets for user {username}: {e}")
            return []

class SocialMediaAnalyzer:
    """Analyze social media content for influencer insights."""
    
    def __init__(self):
        self.reddit_scraper = RedditScraper()
        self.twitter_scraper = TwitterScraper()
        self.data_dir = create_data_dir()
    
    def gather_influencer_chatter(self, query: str, platforms: List[str] = None) -> Dict:
        """Gather influencer chatter across multiple platforms."""
        if platforms is None:
            platforms = ['reddit', 'twitter']
        
        results = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'platforms': {}
        }
        
        if 'reddit' in platforms:
            logger.info("Gathering Reddit content...")
            reddit_posts = self.reddit_scraper.search_subreddits(query)
            results['platforms']['reddit'] = {
                'posts': reddit_posts,
                'count': len(reddit_posts)
            }
        
        if 'twitter' in platforms:
            logger.info("Gathering Twitter content...")
            twitter_tweets = self.twitter_scraper.search_tweets(query)
            results['platforms']['twitter'] = {
                'tweets': twitter_tweets,
                'count': len(twitter_tweets)
            }
        
        return results
    
    def analyze_influencer_impact(self, social_data: Dict) -> Dict:
        """Analyze the impact and reach of influencer content."""
        analysis = {
            'total_content': 0,
            'high_engagement_content': [],
            'top_influencers': [],
            'sentiment_summary': {},
            'trending_topics': []
        }
        
        # Analyze Reddit content
        if 'reddit' in social_data['platforms']:
            reddit_posts = social_data['platforms']['reddit']['posts']
            analysis['total_content'] += len(reddit_posts)
            
            # Find high engagement posts
            high_engagement = [post for post in reddit_posts if post['engagement_score'] > 100]
            analysis['high_engagement_content'].extend(high_engagement)
            
            # Find top authors
            author_engagement = {}
            for post in reddit_posts:
                author = post['author']
                if author not in author_engagement:
                    author_engagement[author] = 0
                author_engagement[author] += post['engagement_score']
            
            top_authors = sorted(author_engagement.items(), key=lambda x: x[1], reverse=True)[:10]
            analysis['top_influencers'].extend([
                {'platform': 'reddit', 'username': author, 'engagement': score}
                for author, score in top_authors
            ])
        
        # Analyze Twitter content
        if 'twitter' in social_data['platforms']:
            twitter_tweets = social_data['platforms']['twitter']['tweets']
            analysis['total_content'] += len(twitter_tweets)
            
            # Find high engagement tweets
            high_engagement = [tweet for tweet in twitter_tweets if tweet['engagement_score'] > 50]
            analysis['high_engagement_content'].extend(high_engagement)
            
            # Find top authors
            author_engagement = {}
            for tweet in twitter_tweets:
                author = tweet['author_username']
                if author and author not in author_engagement:
                    author_engagement[author] = 0
                if author:
                    author_engagement[author] += tweet['engagement_score']
            
            top_authors = sorted(author_engagement.items(), key=lambda x: x[1], reverse=True)[:10]
            analysis['top_influencers'].extend([
                {'platform': 'twitter', 'username': author, 'engagement': score}
                for author, score in top_authors
            ])
        
        return analysis
    
    def save_social_data(self, social_data: Dict, filename: str = None) -> str:
        """Save social media data to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"social_data_{timestamp}.jsonl"
        
        filepath = os.path.join(self.data_dir, filename)
        save_jsonl([social_data], filepath)
        logger.info(f"Saved social data to {filepath}")
        return filepath

def gather_social_chatter(query: str, platforms: List[str] = None) -> Dict:
    """Convenience function to gather social media chatter."""
    analyzer = SocialMediaAnalyzer()
    return analyzer.gather_influencer_chatter(query, platforms)

def analyze_social_impact(social_data: Dict) -> Dict:
    """Convenience function to analyze social media impact."""
    analyzer = SocialMediaAnalyzer()
    return analyzer.analyze_influencer_impact(social_data) 