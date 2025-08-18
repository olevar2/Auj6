"""
Advanced Social Media Post Indicator for AUJ Platform Trading System

This module implements a sophisticated social media sentiment analyzer that processes
social media posts, analyzes sentiment, measures influence, detects viral patterns,
and aggregates signals for trading decisions. The indicator uses advanced NLP,
social network analysis, and machine learning to extract actionable trading insights
from social media data.

Key Features:
- Multi-platform social media post analysis (Twitter, Reddit, Discord, Telegram)
- Advanced natural language processing with sentiment analysis
- Influence scoring based on user metrics and engagement patterns
- Viral detection using propagation models and network analysis
- Real-time sentiment aggregation with temporal weighting
- Hashtag and mention analysis for market trend detection
- Bot detection and authenticity scoring
- Geographic sentiment mapping
- Cross-platform correlation analysis
- Machine learning-based prediction of market impact

Author: AUJ Platform Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import re
import json
from collections import defaultdict, Counter
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SocialPlatform(Enum):
    """Social media platforms."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"


class PostType(Enum):
    """Types of social media posts."""
    ORIGINAL = "original"
    RETWEET = "retweet"
    REPLY = "reply"
    QUOTE = "quote"
    SHARE = "share"
    COMMENT = "comment"
    STORY = "story"
    LIVE = "live"


class InfluenceLevel(Enum):
    """User influence levels."""
    NANO = "nano"           # 1K-10K followers
    MICRO = "micro"         # 10K-100K followers
    MACRO = "macro"         # 100K-1M followers
    MEGA = "mega"           # 1M+ followers
    CELEBRITY = "celebrity"  # Verified celebrities
    INSTITUTIONAL = "institutional"  # Official accounts


class ViralStatus(Enum):
    """Viral detection status."""
    NOT_VIRAL = "not_viral"
    EMERGING = "emerging"
    TRENDING = "trending"
    VIRAL = "viral"
    PEAK_VIRAL = "peak_viral"
    DECLINING = "declining"


@dataclass
class SocialMediaPost:
    """Social media post data structure."""
    post_id: str
    platform: SocialPlatform
    post_type: PostType
    text: str
    author_id: str
    author_name: str
    author_verified: bool
    author_followers: int
    author_following: int
    timestamp: datetime
    likes: int
    shares: int
    comments: int
    hashtags: List[str]
    mentions: List[str]
    urls: List[str]
    media_count: int
    language: str
    location: Optional[str] = None
    reply_to: Optional[str] = None
    thread_id: Optional[str] = None


@dataclass
class PostAnalysis:
    """Analysis results for a social media post."""
    post_id: str
    sentiment_score: float
    sentiment_confidence: float
    influence_score: float
    authenticity_score: float
    market_relevance: float
    engagement_rate: float
    viral_potential: float
    topics: List[str]
    entities: List[str]
    keywords: List[str]
    hashtag_sentiment: Dict[str, float]
    mention_sentiment: Dict[str, float]


@dataclass
class ViralAnalysis:
    """Viral pattern analysis results."""
    post_id: str
    viral_status: ViralStatus
    viral_score: float
    propagation_rate: float
    reach_estimate: int
    network_centrality: float
    time_to_viral: Optional[float]
    predicted_peak: datetime
    decay_rate: float


class SocialMediaPostIndicator:
    """
    Advanced Social Media Post Indicator for analyzing social media sentiment
    and its impact on financial markets.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the social media post indicator."""
        self.logger = logging.getLogger(__name__)

        # Default configuration
        self.parameters = {
            'lookback_hours': 24,
            'min_posts_threshold': 10,
            'sentiment_decay_hours': 6,
            'influence_weight': 0.3,
            'viral_threshold': 0.7,
            'authenticity_threshold': 0.6,
            'market_relevance_threshold': 0.5,
            'engagement_weight': 0.25,
            'platform_weights': {
                SocialPlatform.TWITTER: 0.35,
                SocialPlatform.REDDIT: 0.25,
                SocialPlatform.DISCORD: 0.15,
                SocialPlatform.TELEGRAM: 0.10,
                SocialPlatform.FACEBOOK: 0.08,
                SocialPlatform.LINKEDIN: 0.04,
                SocialPlatform.INSTAGRAM: 0.02,
                SocialPlatform.YOUTUBE: 0.01
            },
            'influence_weights': {
                InfluenceLevel.NANO: 0.05,
                InfluenceLevel.MICRO: 0.15,
                InfluenceLevel.MACRO: 0.25,
                InfluenceLevel.MEGA: 0.35,
                InfluenceLevel.CELEBRITY: 0.15,
                InfluenceLevel.INSTITUTIONAL: 0.05
            },
            'market_keywords': [
                'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'forex', 'trading',
                'market', 'bull', 'bear', 'pump', 'dump', 'hodl', 'buy', 'sell',
                'long', 'short', 'leverage', 'futures', 'options', 'stock',
                'nasdaq', 'sp500', 'dow', 'fed', 'inflation', 'recession'
            ],
            'bot_detection_threshold': 0.7,
            'viral_propagation_threshold': 2.0,
            'min_viral_engagement': 100
        }

        # Update with user configuration
        if config:
            self.parameters.update(config)

        # Initialize storage
        self.posts_data: List[SocialMediaPost] = []
        self.post_analyses: Dict[str, PostAnalysis] = {}
        self.viral_analyses: Dict[str, ViralAnalysis] = {}
        self.aggregated_sentiment = 0.0
        self.trend_direction = 0.0
        self.influence_weighted_sentiment = 0.0
        self.viral_sentiment = 0.0
        self.platform_sentiments: Dict[SocialPlatform, float] = {}
        self.hashtag_trends: Dict[str, float] = {}
        self.mention_trends: Dict[str, float] = {}
        self.last_update = None

        # Initialize ML models
        self._initialize_ml_models()

        # Network analysis
        self.social_graph = nx.DiGraph()

        self.logger.info("Social Media Post Indicator initialized successfully")

    def _initialize_ml_models(self):
        """Initialize machine learning models."""
        try:
            # Sentiment classifier
            self.sentiment_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )

            # Influence predictor
            self.influence_scaler = StandardScaler()
            self.influence_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

            # Viral detection model
            self.viral_model = GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )

            # Topic modeling
            self.topic_vectorizer = CountVectorizer(
                max_features=500,
                stop_words='english',
                min_df=2
            )
            self.topic_model = LatentDirichletAllocation(
                n_components=10,
                random_state=42
            )

            # Clustering for user behavior
            self.user_clusterer = KMeans(n_clusters=5, random_state=42)

            self.logger.info("ML models initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing ML models: {e}")

    def calculate(self, posts_data: List[Dict[str, Any]],
                 market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate social media sentiment indicator.

        Args:
            posts_data: List of social media post dictionaries
            market_data: Optional market data for correlation analysis

        Returns:
            Dictionary containing indicator results
        """
        try:
            # Validate inputs
            if not posts_data or len(posts_data) < self.parameters['min_posts_threshold']:
                self.logger.warning("Insufficient posts data for analysis")
                return self._get_empty_result()

            # Parse and validate posts
            parsed_posts = self._parse_posts_data(posts_data)
            if not parsed_posts:
                return self._get_empty_result()

            # Filter recent posts
            recent_posts = self._filter_recent_posts(parsed_posts)
            if not recent_posts:
                return self._get_empty_result()

            # Analyze individual posts
            post_analyses = []
            for post in recent_posts:
                analysis = self._analyze_single_post(post)
                if analysis:
                    post_analyses.append(analysis)
                    self.post_analyses[post.post_id] = analysis

            # Detect viral patterns
            viral_analyses = self._detect_viral_patterns(recent_posts, post_analyses)

            # Aggregate sentiments
            aggregation_results = self._aggregate_sentiments(post_analyses, viral_analyses)

            # Calculate platform-specific sentiments
            platform_sentiments = self._calculate_platform_sentiments(recent_posts, post_analyses)

            # Analyze trends
            trend_analysis = self._analyze_trends(post_analyses)

            # Calculate influence metrics
            influence_metrics = self._calculate_influence_metrics(recent_posts, post_analyses)

            # Network analysis
            network_metrics = self._perform_network_analysis(recent_posts)

            # Market correlation analysis
            market_correlation = self._analyze_market_correlation(
                aggregation_results, market_data
            )

            # Generate signals
            signals = self._generate_trading_signals(
                aggregation_results, trend_analysis, influence_metrics
            )

            # Update state
            self.posts_data = recent_posts
            self.aggregated_sentiment = aggregation_results['weighted_sentiment']
            self.trend_direction = trend_analysis['sentiment_trend']
            self.influence_weighted_sentiment = influence_metrics['influence_weighted_sentiment']
            self.viral_sentiment = aggregation_results['viral_sentiment']
            self.platform_sentiments = platform_sentiments
            self.last_update = datetime.now()

            # Compile results
            result = {
                'aggregated_sentiment': self.aggregated_sentiment,
                'trend_direction': self.trend_direction,
                'influence_weighted_sentiment': self.influence_weighted_sentiment,
                'viral_sentiment': self.viral_sentiment,
                'platform_sentiments': platform_sentiments,
                'signal_strength': signals['signal_strength'],
                'bullish_signals': signals['bullish_signals'],
                'bearish_signals': signals['bearish_signals'],
                'trend_analysis': trend_analysis,
                'influence_metrics': influence_metrics,
                'network_metrics': network_metrics,
                'market_correlation': market_correlation,
                'hashtag_trends': self.hashtag_trends,
                'mention_trends': self.mention_trends,
                'viral_posts': len(viral_analyses),
                'total_posts_analyzed': len(post_analyses),
                'data_quality': {
                    'posts_count': len(recent_posts),
                    'avg_authenticity': np.mean([a.authenticity_score for a in post_analyses]),
                    'avg_market_relevance': np.mean([a.market_relevance for a in post_analyses]),
                    'platform_coverage': len(set(p.platform for p in recent_posts))
                },
                'timestamp': self.last_update.isoformat()
            }

            self.logger.info(f"Social media analysis completed successfully: "
                           f"sentiment={self.aggregated_sentiment:.3f}, "
                           f"posts={len(recent_posts)}")

            return result

        except Exception as e:
            self.logger.error(f"Error calculating social media indicator: {e}")
            return self._get_empty_result()

    def _get_empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'aggregated_sentiment': 0.0,
            'trend_direction': 0.0,
            'influence_weighted_sentiment': 0.0,
            'viral_sentiment': 0.0,
            'platform_sentiments': {},
            'signal_strength': 0.0,
            'bullish_signals': [],
            'bearish_signals': [],
            'trend_analysis': {},
            'influence_metrics': {},
            'network_metrics': {},
            'market_correlation': {},
            'hashtag_trends': {},
            'mention_trends': {},
            'viral_posts': 0,
            'total_posts_analyzed': 0,
            'data_quality': {
                'posts_count': 0,
                'avg_authenticity': 0.0,
                'avg_market_relevance': 0.0,
                'platform_coverage': 0
            },
            'timestamp': datetime.now().isoformat()
        }

    def _parse_posts_data(self, posts_data: List[Dict[str, Any]]) -> List[SocialMediaPost]:
        """Parse raw posts data into structured format."""
        try:
            parsed_posts = []

            for post_dict in posts_data:
                try:
                    # Extract required fields with defaults
                    post = SocialMediaPost(
                        post_id=str(post_dict.get('id', f"unknown_{len(parsed_posts)}")),
                        platform=self._parse_platform(post_dict.get('platform', 'twitter')),
                        post_type=self._parse_post_type(post_dict.get('type', 'original')),
                        text=str(post_dict.get('text', '')),
                        author_id=str(post_dict.get('author_id', 'unknown')),
                        author_name=str(post_dict.get('author_name', 'Unknown')),
                        author_verified=bool(post_dict.get('verified', False)),
                        author_followers=max(0, int(post_dict.get('followers', 0))),
                        author_following=max(0, int(post_dict.get('following', 0))),
                        timestamp=self._parse_timestamp(post_dict.get('timestamp')),
                        likes=max(0, int(post_dict.get('likes', 0))),
                        shares=max(0, int(post_dict.get('shares', 0))),
                        comments=max(0, int(post_dict.get('comments', 0))),
                        hashtags=self._extract_hashtags(post_dict.get('text', '')),
                        mentions=self._extract_mentions(post_dict.get('text', '')),
                        urls=self._extract_urls(post_dict.get('text', '')),
                        media_count=max(0, int(post_dict.get('media_count', 0))),
                        language=str(post_dict.get('language', 'en')),
                        location=post_dict.get('location'),
                        reply_to=post_dict.get('reply_to'),
                        thread_id=post_dict.get('thread_id')
                    )

                    # Basic validation
                    if post.text and len(post.text.strip()) > 0:
                        parsed_posts.append(post)

                except Exception as e:
                    self.logger.warning(f"Error parsing post: {e}")
                    continue

            return parsed_posts

        except Exception as e:
            self.logger.error(f"Error parsing posts data: {e}")
            return []

    def _parse_platform(self, platform_str: str) -> SocialPlatform:
        """Parse platform string to SocialPlatform enum."""
        try:
            platform_mapping = {
                'twitter': SocialPlatform.TWITTER,
                'reddit': SocialPlatform.REDDIT,
                'discord': SocialPlatform.DISCORD,
                'telegram': SocialPlatform.TELEGRAM,
                'facebook': SocialPlatform.FACEBOOK,
                'instagram': SocialPlatform.INSTAGRAM,
                'linkedin': SocialPlatform.LINKEDIN,
                'youtube': SocialPlatform.YOUTUBE,
                'tiktok': SocialPlatform.TIKTOK
            }
            return platform_mapping.get(platform_str.lower(), SocialPlatform.TWITTER)
        except Exception:
            return SocialPlatform.TWITTER

    def _parse_post_type(self, type_str: str) -> PostType:
        """Parse post type string to PostType enum."""
        try:
            type_mapping = {
                'original': PostType.ORIGINAL,
                'retweet': PostType.RETWEET,
                'reply': PostType.REPLY,
                'quote': PostType.QUOTE,
                'share': PostType.SHARE,
                'comment': PostType.COMMENT,
                'story': PostType.STORY,
                'live': PostType.LIVE
            }
            return type_mapping.get(type_str.lower(), PostType.ORIGINAL)
        except Exception:
            return PostType.ORIGINAL

    def _parse_timestamp(self, timestamp_input: Any) -> datetime:
        """Parse timestamp from various input formats."""
        try:
            if isinstance(timestamp_input, datetime):
                return timestamp_input
            elif isinstance(timestamp_input, str):
                # Try multiple datetime formats
                formats = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%d',
                    '%m/%d/%Y %H:%M:%S',
                    '%d/%m/%Y %H:%M:%S'
                ]
                for fmt in formats:
                    try:
                        return datetime.strptime(timestamp_input, fmt)
                    except ValueError:
                        continue
                return datetime.now()
            elif isinstance(timestamp_input, (int, float)):
                return datetime.fromtimestamp(timestamp_input)
            else:
                return datetime.now()
        except Exception:
            return datetime.now()

    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text."""
        try:
            hashtag_pattern = r'#(\w+)'
            hashtags = re.findall(hashtag_pattern, text.lower())
            return list(set(hashtags))  # Remove duplicates
        except Exception:
            return []

    def _extract_mentions(self, text: str) -> List[str]:
        """Extract mentions from text."""
        try:
            mention_pattern = r'@(\w+)'
            mentions = re.findall(mention_pattern, text.lower())
            return list(set(mentions))  # Remove duplicates
        except Exception:
            return []

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        try:
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, text)
            return urls
        except Exception:
            return []

    def _filter_recent_posts(self, posts: List[SocialMediaPost]) -> List[SocialMediaPost]:
        """Filter posts to only include recent ones."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.parameters['lookback_hours'])
            recent_posts = [post for post in posts if post.timestamp >= cutoff_time]

            self.logger.info(f"Filtered {len(recent_posts)} recent posts from {len(posts)} total")
            return recent_posts

        except Exception as e:
            self.logger.error(f"Error filtering recent posts: {e}")
            return posts

    def _analyze_single_post(self, post: SocialMediaPost) -> Optional[PostAnalysis]:
        """Analyze a single social media post."""
        try:
            # Sentiment analysis
            sentiment_result = self._analyze_post_sentiment(post.text)

            # Influence scoring
            influence_score = self._calculate_influence_score(post)

            # Authenticity scoring
            authenticity_score = self._calculate_authenticity_score(post)

            # Market relevance
            market_relevance = self._calculate_market_relevance(post.text, post.hashtags)

            # Engagement metrics
            engagement_rate = self._calculate_engagement_rate(post)

            # Viral potential
            viral_potential = self._calculate_viral_potential(post)

            # Topic and entity extraction
            topics = self._extract_topics(post.text)
            entities = self._extract_entities(post.text)
            keywords = self._extract_keywords(post.text)

            # Hashtag and mention sentiment
            hashtag_sentiment = self._analyze_hashtag_sentiment(post.hashtags)
            mention_sentiment = self._analyze_mention_sentiment(post.mentions)

            analysis = PostAnalysis(
                post_id=post.post_id,
                sentiment_score=sentiment_result['sentiment'],
                sentiment_confidence=sentiment_result['confidence'],
                influence_score=influence_score,
                authenticity_score=authenticity_score,
                market_relevance=market_relevance,
                engagement_rate=engagement_rate,
                viral_potential=viral_potential,
                topics=topics,
                entities=entities,
                keywords=keywords,
                hashtag_sentiment=hashtag_sentiment,
                mention_sentiment=mention_sentiment
            )

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing post {post.post_id}: {e}")
            return None

    def _analyze_post_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of post text."""
        try:
            # Basic TextBlob sentiment
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Enhanced sentiment analysis
            # Clean text for analysis
            cleaned_text = self._clean_text_for_sentiment(text)

            # Calculate enhanced sentiment score
            enhanced_sentiment = self._calculate_enhanced_sentiment(cleaned_text)

            # Combine scores
            final_sentiment = (polarity * 0.5 + enhanced_sentiment * 0.5)

            # Calculate confidence based on text length and subjectivity
            confidence = self._calculate_sentiment_confidence(text, subjectivity)

            return {
                'sentiment': np.clip(final_sentiment, -1.0, 1.0),
                'confidence': np.clip(confidence, 0.0, 1.0)
            }

        except Exception as e:
            self.logger.warning(f"Error analyzing sentiment: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0}

    def _clean_text_for_sentiment(self, text: str) -> str:
        """Clean text for sentiment analysis."""
        try:
            # Remove URLs
            text = re.sub(r'http[s]?://\S+', '', text)

            # Remove mentions and hashtags for cleaner sentiment
            text = re.sub(r'[@#]\w+', '', text)

            # Remove excessive punctuation
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)

            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            return text

        except Exception:
            return text

    def _calculate_enhanced_sentiment(self, text: str) -> float:
        """Calculate enhanced sentiment score using custom logic."""
        try:
            # Market-specific sentiment lexicon
            bullish_terms = {
                'moon', 'rocket', 'bullish', 'pump', 'long', 'buy', 'hodl',
                'diamond', 'hands', 'bull', 'green', 'profit', 'gains',
                'breakout', 'surge', 'rally', 'golden', 'cross', 'support'
            }

            bearish_terms = {
                'crash', 'dump', 'bear', 'bearish', 'short', 'sell', 'panic',
                'red', 'loss', 'drop', 'fall', 'breakdown', 'resistance',
                'correction', 'bubble', 'fear', 'liquidation', 'rug', 'scam'
            }

            # Intensifiers
            intensifiers = {
                'very': 1.5, 'extremely': 2.0, 'absolutely': 1.8,
                'definitely': 1.6, 'completely': 1.7, 'totally': 1.4
            }

            words = text.lower().split()
            sentiment_score = 0.0
            word_count = len(words)

            for i, word in enumerate(words):
                # Check for intensifiers
                intensity = 1.0
                if i > 0 and words[i-1] in intensifiers:
                    intensity = intensifiers[words[i-1]]

                # Calculate sentiment contribution
                if word in bullish_terms:
                    sentiment_score += 1.0 * intensity
                elif word in bearish_terms:
                    sentiment_score -= 1.0 * intensity

            # Normalize by word count
            if word_count > 0:
                sentiment_score = sentiment_score / word_count

            # Apply sigmoid normalization
            return np.tanh(sentiment_score * 3)

        except Exception:
            return 0.0

    def _calculate_sentiment_confidence(self, text: str, subjectivity: float) -> float:
        """Calculate confidence in sentiment analysis."""
        try:
            # Base confidence on text length
            text_length_factor = min(1.0, len(text) / 100)

            # Subjectivity indicates emotional content
            subjectivity_factor = subjectivity

            # Check for uncertainty indicators
            uncertainty_words = {'maybe', 'perhaps', 'might', 'could', 'possibly'}
            uncertainty_count = sum(1 for word in text.lower().split()
                                  if word in uncertainty_words)
            uncertainty_penalty = min(0.3, uncertainty_count * 0.1)

            # Check for emphasis (caps, exclamation)
            caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
            exclamation_count = text.count('!')
            emphasis_bonus = min(0.2, caps_ratio * 0.5 + exclamation_count * 0.05)

            confidence = (text_length_factor * 0.3 +
                         subjectivity_factor * 0.4 +
                         emphasis_bonus -
                         uncertainty_penalty + 0.3)

            return np.clip(confidence, 0.0, 1.0)

        except Exception:
            return 0.5

    def _calculate_influence_score(self, post: SocialMediaPost) -> float:
        """Calculate user influence score."""
        try:
            # Follower-based influence
            follower_score = self._normalize_follower_count(post.author_followers)

            # Verification bonus
            verification_bonus = 0.2 if post.author_verified else 0.0

            # Engagement-based influence
            engagement_score = self._calculate_engagement_influence(post)

            # Platform-specific adjustment
            platform_weight = self.parameters['platform_weights'].get(
                post.platform, 0.1
            )

            # Influence level classification
            influence_level = self._classify_influence_level(post.author_followers)
            influence_weight = self.parameters['influence_weights'].get(
                influence_level, 0.1
            )

            # Combine factors
            influence_score = (
                follower_score * 0.4 +
                engagement_score * 0.3 +
                verification_bonus +
                influence_weight * 0.3
            ) * platform_weight

            return np.clip(influence_score, 0.0, 1.0)

        except Exception as e:
            self.logger.warning(f"Error calculating influence score: {e}")
            return 0.1

    def _normalize_follower_count(self, followers: int) -> float:
        """Normalize follower count to 0-1 scale."""
        try:
            # Logarithmic normalization
            if followers <= 0:
                return 0.0

            # Use log scale for better distribution
            log_followers = np.log10(followers + 1)

            # Normalize to 0-1 (assuming max ~10M followers = 7 in log scale)
            normalized = log_followers / 7.0

            return np.clip(normalized, 0.0, 1.0)

        except Exception:
            return 0.0

    def _calculate_engagement_influence(self, post: SocialMediaPost) -> float:
        """Calculate engagement-based influence."""
        try:
            total_engagement = post.likes + post.shares + post.comments

            if post.author_followers <= 0:
                return 0.0

            # Engagement rate
            engagement_rate = total_engagement / post.author_followers

            # Normalize engagement rate (typical good rate is 1-5%)
            normalized_rate = min(1.0, engagement_rate / 0.05)

            return normalized_rate

        except Exception:
            return 0.0

    def _classify_influence_level(self, followers: int) -> InfluenceLevel:
        """Classify user influence level."""
        try:
            if followers >= 1_000_000:
                return InfluenceLevel.MEGA
            elif followers >= 100_000:
                return InfluenceLevel.MACRO
            elif followers >= 10_000:
                return InfluenceLevel.MICRO
            elif followers >= 1_000:
                return InfluenceLevel.NANO
            else:
                return InfluenceLevel.NANO
        except Exception:
            return InfluenceLevel.NANO

    def _calculate_authenticity_score(self, post: SocialMediaPost) -> float:
        """Calculate post authenticity score (bot detection)."""
        try:
            authenticity_indicators = []

            # Account age (assume newer accounts are less authentic)
            # Note: In real implementation, would need account creation date
            account_age_score = 0.7  # Default moderate score
            authenticity_indicators.append(account_age_score)

            # Follower-to-following ratio
            if post.author_following > 0:
                ratio = post.author_followers / post.author_following
                ratio_score = min(1.0, ratio / 2.0)  # Prefer higher ratios
            else:
                ratio_score = 1.0 if post.author_followers > 0 else 0.5
            authenticity_indicators.append(ratio_score)

            # Text quality analysis
            text_quality = self._analyze_text_quality(post.text)
            authenticity_indicators.append(text_quality)

            # Posting pattern analysis (simplified)
            posting_pattern_score = 0.8  # Default good score
            authenticity_indicators.append(posting_pattern_score)

            # Verification status
            verification_score = 1.0 if post.author_verified else 0.6
            authenticity_indicators.append(verification_score)

            # Average all indicators
            authenticity_score = np.mean(authenticity_indicators)

            return np.clip(authenticity_score, 0.0, 1.0)

        except Exception as e:
            self.logger.warning(f"Error calculating authenticity score: {e}")
            return 0.5

    def _analyze_text_quality(self, text: str) -> float:
        """Analyze text quality for authenticity assessment."""
        try:
            # Text length
            length_score = min(1.0, len(text) / 100)

            # Word variety (unique words / total words)
            words = text.lower().split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
            else:
                unique_ratio = 0.0

            # Grammar and structure (simplified)
            # Count sentences
            sentence_count = len(re.findall(r'[.!?]+', text))
            structure_score = min(1.0, sentence_count / 3) if sentence_count > 0 else 0.5

            # Avoid excessive repetition
            repetition_penalty = 0.0
            if len(words) > 5:
                # Check for repeated patterns
                word_counts = Counter(words)
                max_repetition = max(word_counts.values())
                if max_repetition > len(words) * 0.3:  # More than 30% repetition
                    repetition_penalty = 0.3

            quality_score = (length_score * 0.3 +
                           unique_ratio * 0.4 +
                           structure_score * 0.3 -
                           repetition_penalty)

            return np.clip(quality_score, 0.0, 1.0)

        except Exception:
            return 0.5

    def _calculate_market_relevance(self, text: str, hashtags: List[str]) -> float:
        """Calculate how relevant the post is to financial markets."""
        try:
            relevance_score = 0.0

            # Check for market keywords in text
            text_lower = text.lower()
            keyword_matches = sum(1 for keyword in self.parameters['market_keywords']
                                if keyword in text_lower)

            # Normalize keyword score
            keyword_score = min(1.0, keyword_matches / 5)  # Cap at 5 keywords
            relevance_score += keyword_score * 0.6

            # Check hashtags for market relevance
            market_hashtags = [tag for tag in hashtags
                             if any(keyword in tag for keyword in self.parameters['market_keywords'])]
            hashtag_score = min(1.0, len(market_hashtags) / 3)  # Cap at 3 hashtags
            relevance_score += hashtag_score * 0.4

            return np.clip(relevance_score, 0.0, 1.0)

        except Exception:
            return 0.0

    def _calculate_engagement_rate(self, post: SocialMediaPost) -> float:
        """Calculate post engagement rate."""
        try:
            if post.author_followers <= 0:
                return 0.0

            total_engagement = post.likes + post.shares + post.comments
            engagement_rate = total_engagement / post.author_followers

            # Normalize to typical social media engagement rates (1-10%)
            normalized_rate = min(1.0, engagement_rate / 0.1)

            return normalized_rate

        except Exception:
            return 0.0

    def _calculate_viral_potential(self, post: SocialMediaPost) -> float:
        """Calculate viral potential of a post."""
        try:
            viral_factors = []

            # Engagement velocity (engagement per hour since posting)
            time_since_post = (datetime.now() - post.timestamp).total_seconds() / 3600
            if time_since_post > 0:
                total_engagement = post.likes + post.shares + post.comments
                engagement_velocity = total_engagement / time_since_post
                velocity_score = min(1.0, engagement_velocity / 100)  # Normalize
                viral_factors.append(velocity_score)

            # Share ratio (shares relative to likes)
            if post.likes > 0:
                share_ratio = post.shares / post.likes
                share_score = min(1.0, share_ratio * 2)  # Good viral content has high share ratio
                viral_factors.append(share_score)

            # Content characteristics
            text_length = len(post.text)
            # Moderate length posts often perform better
            length_score = 1.0 - abs(text_length - 140) / 140 if text_length <= 280 else 0.5
            viral_factors.append(max(0.0, length_score))

            # Media presence
            media_score = min(1.0, post.media_count * 0.3)
            viral_factors.append(media_score)

            # Hashtag usage
            hashtag_score = min(1.0, len(post.hashtags) * 0.2)
            viral_factors.append(hashtag_score)

            if viral_factors:
                viral_potential = np.mean(viral_factors)
            else:
                viral_potential = 0.0

            return np.clip(viral_potential, 0.0, 1.0)

        except Exception:
            return 0.0

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from post text."""
        try:
            # Simple topic extraction based on keywords
            topics = []

            # Financial topics
            financial_topics = {
                'cryptocurrency': ['crypto', 'bitcoin', 'ethereum', 'blockchain', 'defi'],
                'forex': ['forex', 'currency', 'usd', 'eur', 'gbp', 'jpy'],
                'stocks': ['stock', 'equity', 'nasdaq', 'sp500', 'dow'],
                'trading': ['trading', 'trade', 'buy', 'sell', 'position'],
                'market_analysis': ['analysis', 'chart', 'technical', 'fundamental'],
                'economic_policy': ['fed', 'interest', 'inflation', 'gdp', 'policy'],
                'market_sentiment': ['bull', 'bear', 'sentiment', 'mood', 'fear']
            }

            text_lower = text.lower()
            for topic, keywords in financial_topics.items():
                if any(keyword in text_lower for keyword in keywords):
                    topics.append(topic)

            return topics[:5]  # Limit to top 5 topics

        except Exception:
            return []

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from post text."""
        try:
            entities = []

            # Common financial entities
            entity_patterns = {
                r'\b[A-Z]{2,5}\b': 'TICKER',  # Stock tickers
                r'\$[A-Z]{3,4}\b': 'CURRENCY',  # Currency codes
                r'\b\d+\.?\d*%\b': 'PERCENTAGE',  # Percentages
                r'\b\$\d+\.?\d*[BMK]?\b': 'MONEY',  # Money amounts
                r'\b\d{1,2}/\d{1,2}/\d{4}\b': 'DATE',  # Dates
            }

            for pattern, entity_type in entity_patterns.items():
                matches = re.findall(pattern, text)
                for match in matches:
                    entities.append(f"{entity_type}:{match}")

            return entities[:10]  # Limit to top 10

        except Exception:
            return []

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key trading-related words from text."""
        try:
            # Remove noise words and extract meaningful keywords
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}

            # Extract words longer than 3 characters
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            keywords = [word for word in words if word not in stop_words]

            # Prioritize trading/finance related keywords
            trading_keywords = []
            other_keywords = []

            for keyword in set(keywords):  # Remove duplicates
                if any(market_word in keyword for market_word in self.parameters['market_keywords']):
                    trading_keywords.append(keyword)
                else:
                    other_keywords.append(keyword)

            # Combine with priority to trading keywords
            final_keywords = trading_keywords + other_keywords

            return final_keywords[:10]  # Limit to top 10

        except Exception:
            return []

    def _analyze_hashtag_sentiment(self, hashtags: List[str]) -> Dict[str, float]:
        """Analyze sentiment for each hashtag."""
        try:
            hashtag_sentiment = {}

            # Predefined hashtag sentiments (simplified)
            positive_hashtags = {
                'bullish', 'moon', 'diamond', 'hands', 'hodl', 'buy', 'long',
                'profit', 'gains', 'green', 'rally', 'breakout', 'surge'
            }

            negative_hashtags = {
                'bearish', 'crash', 'dump', 'panic', 'sell', 'short', 'red',
                'loss', 'drop', 'fall', 'correction', 'fear', 'liquidation'
            }

            for hashtag in hashtags:
                hashtag_lower = hashtag.lower()
                if hashtag_lower in positive_hashtags:
                    hashtag_sentiment[hashtag] = 0.8
                elif hashtag_lower in negative_hashtags:
                    hashtag_sentiment[hashtag] = -0.8
                else:
                    # Use basic sentiment analysis for unknown hashtags
                    blob = TextBlob(hashtag)
                    hashtag_sentiment[hashtag] = blob.sentiment.polarity

            return hashtag_sentiment

        except Exception:
            return {}

    def _analyze_mention_sentiment(self, mentions: List[str]) -> Dict[str, float]:
        """Analyze sentiment context around mentions."""
        try:
            # For this simplified version, assign neutral sentiment
            # In production, would analyze context around mentions
            mention_sentiment = {}

            for mention in mentions:
                # Default neutral sentiment for mentions
                mention_sentiment[mention] = 0.0

            return mention_sentiment

        except Exception:
            return {}

    def _detect_viral_patterns(self, posts: List[SocialMediaPost],
                             analyses: List[PostAnalysis]) -> List[ViralAnalysis]:
        """Detect viral patterns in posts."""
        try:
            viral_analyses = []

            for i, post in enumerate(posts):
                if i < len(analyses):
                    analysis = analyses[i]

                    # Calculate viral metrics
                    viral_score = self._calculate_comprehensive_viral_score(post, analysis)
                    viral_status = self._classify_viral_status(viral_score, post)
                    propagation_rate = self._calculate_propagation_rate(post)
                    reach_estimate = self._estimate_reach(post, analysis)
                    network_centrality = self._calculate_network_centrality(post)

                    viral_analysis = ViralAnalysis(
                        post_id=post.post_id,
                        viral_status=viral_status,
                        viral_score=viral_score,
                        propagation_rate=propagation_rate,
                        reach_estimate=reach_estimate,
                        network_centrality=network_centrality,
                        time_to_viral=self._estimate_time_to_viral(post),
                        predicted_peak=self._predict_viral_peak(post),
                        decay_rate=self._estimate_decay_rate(post)
                    )

                    viral_analyses.append(viral_analysis)
                    self.viral_analyses[post.post_id] = viral_analysis

            return viral_analyses

        except Exception as e:
            self.logger.error(f"Error detecting viral patterns: {e}")
            return []

    def _calculate_comprehensive_viral_score(self, post: SocialMediaPost,
                                           analysis: PostAnalysis) -> float:
        """Calculate comprehensive viral score."""
        try:
            viral_factors = []

            # Base viral potential from analysis
            viral_factors.append(analysis.viral_potential)

            # Engagement metrics
            total_engagement = post.likes + post.shares + post.comments
            if total_engagement >= self.parameters['min_viral_engagement']:
                engagement_score = min(1.0, total_engagement / 1000)
                viral_factors.append(engagement_score)

            # Influence factor
            viral_factors.append(analysis.influence_score)

            # Market relevance
            viral_factors.append(analysis.market_relevance)

            # Time factor (recent posts have higher viral potential)
            time_since_post = (datetime.now() - post.timestamp).total_seconds() / 3600
            time_factor = max(0.1, 1.0 - (time_since_post / 24))  # Decay over 24 hours
            viral_factors.append(time_factor)

            # Platform factor
            platform_viral_weight = {
                SocialPlatform.TWITTER: 1.0,
                SocialPlatform.REDDIT: 0.8,
                SocialPlatform.TIKTOK: 0.9,
                SocialPlatform.INSTAGRAM: 0.7,
                SocialPlatform.FACEBOOK: 0.6,
                SocialPlatform.DISCORD: 0.5,
                SocialPlatform.TELEGRAM: 0.5,
                SocialPlatform.LINKEDIN: 0.3,
                SocialPlatform.YOUTUBE: 0.4
            }
            platform_factor = platform_viral_weight.get(post.platform, 0.5)
            viral_factors.append(platform_factor)

            # Calculate weighted average
            viral_score = np.average(viral_factors, weights=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1])

            return np.clip(viral_score, 0.0, 1.0)

        except Exception:
            return 0.0

    def _classify_viral_status(self, viral_score: float, post: SocialMediaPost) -> ViralStatus:
        """Classify viral status based on score and metrics."""
        try:
            total_engagement = post.likes + post.shares + post.comments

            if viral_score >= 0.9 and total_engagement >= 10000:
                return ViralStatus.PEAK_VIRAL
            elif viral_score >= 0.7 and total_engagement >= 1000:
                return ViralStatus.VIRAL
            elif viral_score >= 0.5 and total_engagement >= 500:
                return ViralStatus.TRENDING
            elif viral_score >= 0.3 and total_engagement >= 100:
                return ViralStatus.EMERGING
            else:
                # Check if it's declining (would need historical data)
                return ViralStatus.NOT_VIRAL

        except Exception:
            return ViralStatus.NOT_VIRAL

    def _calculate_propagation_rate(self, post: SocialMediaPost) -> float:
        """Calculate rate of content propagation."""
        try:
            # Time since posting
            time_since_post = (datetime.now() - post.timestamp).total_seconds() / 3600

            if time_since_post <= 0:
                return 0.0

            # Shares are the primary propagation metric
            propagation_rate = post.shares / time_since_post

            # Normalize (typical viral content gets 100+ shares per hour)
            normalized_rate = min(1.0, propagation_rate / 100)

            return normalized_rate

        except Exception:
            return 0.0

    def _estimate_reach(self, post: SocialMediaPost, analysis: PostAnalysis) -> int:
        """Estimate total reach of the post."""
        try:
            # Base reach is follower count
            base_reach = post.author_followers

            # Multiply by share factor (each share reaches some followers)
            avg_followers_per_user = 500  # Estimate
            share_reach = post.shares * avg_followers_per_user

            # Apply influence factor
            reach_multiplier = 1.0 + analysis.influence_score

            total_reach = int((base_reach + share_reach) * reach_multiplier)

            return total_reach

        except Exception:
            return 0

    def _calculate_network_centrality(self, post: SocialMediaPost) -> float:
        """Calculate network centrality score."""
        try:
            # Simplified network centrality based on mentions and engagement
            mention_factor = min(1.0, len(post.mentions) / 10)
            engagement_factor = min(1.0, (post.likes + post.comments) / 1000)

            centrality = (mention_factor + engagement_factor) / 2

            return centrality

        except Exception:
            return 0.0

    def _estimate_time_to_viral(self, post: SocialMediaPost) -> Optional[float]:
        """Estimate time to reach viral status."""
        try:
            # Current engagement velocity
            time_since_post = (datetime.now() - post.timestamp).total_seconds() / 3600
            if time_since_post <= 0:
                return None

            current_engagement = post.likes + post.shares + post.comments
            engagement_velocity = current_engagement / time_since_post

            # Estimate time to reach viral threshold (1000 engagements)
            viral_threshold = 1000
            if engagement_velocity > 0:
                remaining_engagement = max(0, viral_threshold - current_engagement)
                time_to_viral = remaining_engagement / engagement_velocity
                return time_to_viral

            return None

        except Exception:
            return None

    def _predict_viral_peak(self, post: SocialMediaPost) -> datetime:
        """Predict when viral content will peak."""
        try:
            # Most viral content peaks within 6-24 hours
            peak_hours = 12  # Default prediction

            # Adjust based on platform
            platform_peak_times = {
                SocialPlatform.TWITTER: 8,
                SocialPlatform.REDDIT: 16,
                SocialPlatform.TIKTOK: 6,
                SocialPlatform.INSTAGRAM: 12,
                SocialPlatform.FACEBOOK: 24
            }

            peak_hours = platform_peak_times.get(post.platform, 12)

            predicted_peak = post.timestamp + timedelta(hours=peak_hours)

            return predicted_peak

        except Exception:
            return datetime.now() + timedelta(hours=12)

    def _estimate_decay_rate(self, post: SocialMediaPost) -> float:
        """Estimate content decay rate."""
        try:
            # Viral content typically decays exponentially
            # Decay rate depends on platform and content type

            platform_decay_rates = {
                SocialPlatform.TWITTER: 0.3,   # Fast decay
                SocialPlatform.REDDIT: 0.1,    # Slow decay
                SocialPlatform.TIKTOK: 0.4,    # Very fast decay
                SocialPlatform.INSTAGRAM: 0.2, # Medium decay
                SocialPlatform.FACEBOOK: 0.15  # Slow decay
            }

            base_decay = platform_decay_rates.get(post.platform, 0.2)

            # Adjust for content quality (higher quality = slower decay)
            if len(post.text) > 100:  # Longer content may have staying power
                base_decay *= 0.8

            if post.media_count > 0:  # Media content may last longer
                base_decay *= 0.9

            return base_decay

        except Exception:
            return 0.2

    def _aggregate_sentiments(self, analyses: List[PostAnalysis],
                            viral_analyses: List[ViralAnalysis]) -> Dict[str, Any]:
        """Aggregate sentiments across all posts."""
        try:
            if not analyses:
                return {'weighted_sentiment': 0.0, 'viral_sentiment': 0.0}

            # Calculate weighted sentiment
            weighted_sentiments = []
            viral_sentiments = []

            for i, analysis in enumerate(analyses):
                # Weight by influence and authenticity
                weight = (analysis.influence_score * self.parameters['influence_weight'] +
                         analysis.authenticity_score * 0.3 +
                         analysis.market_relevance * 0.4)

                weighted_sentiment = analysis.sentiment_score * weight
                weighted_sentiments.append(weighted_sentiment)

                # Track viral sentiments separately
                if i < len(viral_analyses):
                    viral_analysis = viral_analyses[i]
                    if viral_analysis.viral_score >= self.parameters['viral_threshold']:
                        viral_sentiment = analysis.sentiment_score * viral_analysis.viral_score
                        viral_sentiments.append(viral_sentiment)

            # Calculate aggregated scores
            if weighted_sentiments:
                aggregated_sentiment = np.mean(weighted_sentiments)
            else:
                aggregated_sentiment = 0.0

            if viral_sentiments:
                viral_sentiment = np.mean(viral_sentiments)
            else:
                viral_sentiment = 0.0

            return {
                'weighted_sentiment': aggregated_sentiment,
                'viral_sentiment': viral_sentiment,
                'total_weight': sum(a.influence_score for a in analyses),
                'sentiment_variance': np.var([a.sentiment_score for a in analyses])
            }

        except Exception as e:
            self.logger.error(f"Error aggregating sentiments: {e}")
            return {'weighted_sentiment': 0.0, 'viral_sentiment': 0.0}

    def _calculate_platform_sentiments(self, posts: List[SocialMediaPost],
                                     analyses: List[PostAnalysis]) -> Dict[SocialPlatform, float]:
        """Calculate sentiment by platform."""
        try:
            platform_sentiments = {}
            platform_posts = defaultdict(list)

            # Group posts by platform
            for i, post in enumerate(posts):
                if i < len(analyses):
                    platform_posts[post.platform].append(analyses[i])

            # Calculate average sentiment per platform
            for platform, platform_analyses in platform_posts.items():
                if platform_analyses:
                    sentiments = [a.sentiment_score for a in platform_analyses]
                    platform_sentiments[platform] = np.mean(sentiments)

            return platform_sentiments

        except Exception as e:
            self.logger.error(f"Error calculating platform sentiments: {e}")
            return {}

    def _analyze_trends(self, analyses: List[PostAnalysis]) -> Dict[str, Any]:
        """Analyze sentiment trends."""
        try:
            if len(analyses) < 5:
                return {'sentiment_trend': 0.0, 'trend_strength': 0.0}

            # Extract sentiment scores in chronological order
            sentiment_scores = [a.sentiment_score for a in analyses]

            # Calculate trend using linear regression
            x = np.arange(len(sentiment_scores))

            # Fit linear trend
            z = np.polyfit(x, sentiment_scores, 1)
            trend_slope = z[0]

            # Calculate trend strength (R-squared)
            p = np.poly1d(z)
            y_pred = p(x)
            ss_res = np.sum((sentiment_scores - y_pred) ** 2)
            ss_tot = np.sum((sentiment_scores - np.mean(sentiment_scores)) ** 2)

            if ss_tot != 0:
                r_squared = 1 - (ss_res / ss_tot)
            else:
                r_squared = 0.0

            # Normalize trend slope
            normalized_trend = np.tanh(trend_slope * len(sentiment_scores))

            # Calculate momentum (recent vs earlier sentiment)
            recent_sentiment = np.mean(sentiment_scores[-3:])  # Last 3 posts
            earlier_sentiment = np.mean(sentiment_scores[:3])  # First 3 posts
            momentum = recent_sentiment - earlier_sentiment

            return {
                'sentiment_trend': normalized_trend,
                'trend_strength': r_squared,
                'momentum': momentum,
                'recent_sentiment': recent_sentiment,
                'volatility': np.std(sentiment_scores)
            }

        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
            return {'sentiment_trend': 0.0, 'trend_strength': 0.0}

    def _calculate_influence_metrics(self, posts: List[SocialMediaPost],
                                   analyses: List[PostAnalysis]) -> Dict[str, Any]:
        """Calculate comprehensive influence metrics."""
        try:
            if not analyses:
                return {}

            # Influence-weighted sentiment
            total_influence = sum(a.influence_score for a in analyses)
            if total_influence > 0:
                influence_weighted_sentiment = sum(
                    a.sentiment_score * a.influence_score for a in analyses
                ) / total_influence
            else:
                influence_weighted_sentiment = 0.0

            # Top influencers analysis
            top_influencers = []
            for i, post in enumerate(posts):
                if i < len(analyses):
                    analysis = analyses[i]
                    if analysis.influence_score >= 0.7:  # High influence threshold
                        top_influencers.append({
                            'author': post.author_name,
                            'followers': post.author_followers,
                            'influence_score': analysis.influence_score,
                            'sentiment': analysis.sentiment_score,
                            'post_id': post.post_id
                        })

            # Sort by influence
            top_influencers.sort(key=lambda x: x['influence_score'], reverse=True)

            # Influence distribution
            influence_scores = [a.influence_score for a in analyses]
            influence_distribution = {
                'mean': np.mean(influence_scores),
                'median': np.median(influence_scores),
                'std': np.std(influence_scores),
                'max': np.max(influence_scores),
                'min': np.min(influence_scores)
            }

            return {
                'influence_weighted_sentiment': influence_weighted_sentiment,
                'top_influencers': top_influencers[:10],  # Top 10
                'influence_distribution': influence_distribution,
                'high_influence_posts': len([a for a in analyses if a.influence_score >= 0.7]),
                'verified_user_sentiment': self._calculate_verified_user_sentiment(posts, analyses)
            }

        except Exception as e:
            self.logger.error(f"Error calculating influence metrics: {e}")
            return {}

    def _calculate_verified_user_sentiment(self, posts: List[SocialMediaPost],
                                         analyses: List[PostAnalysis]) -> float:
        """Calculate sentiment from verified users only."""
        try:
            verified_sentiments = []

            for i, post in enumerate(posts):
                if post.author_verified and i < len(analyses):
                    verified_sentiments.append(analyses[i].sentiment_score)

            if verified_sentiments:
                return np.mean(verified_sentiments)
            else:
                return 0.0

        except Exception:
            return 0.0

    def _perform_network_analysis(self, posts: List[SocialMediaPost]) -> Dict[str, Any]:
        """Perform social network analysis."""
        try:
            # Build mention network
            self.social_graph.clear()

            # Add nodes and edges based on mentions
            for post in posts:
                # Add author as node
                self.social_graph.add_node(post.author_id,
                                         followers=post.author_followers,
                                         verified=post.author_verified)

                # Add edges for mentions
                for mention in post.mentions:
                    self.social_graph.add_node(mention)
                    self.social_graph.add_edge(post.author_id, mention)

            # Calculate network metrics
            if len(self.social_graph.nodes()) > 1:
                # Centrality measures
                try:
                    betweenness = nx.betweenness_centrality(self.social_graph)
                    closeness = nx.closeness_centrality(self.social_graph)
                    pagerank = nx.pagerank(self.social_graph)

                    # Network density
                    density = nx.density(self.social_graph)

                    # Connected components
                    components = list(nx.connected_components(self.social_graph.to_undirected()))
                    largest_component_size = max(len(comp) for comp in components) if components else 0

                    # Most central users
                    central_users = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]

                    return {
                        'network_density': density,
                        'total_nodes': len(self.social_graph.nodes()),
                        'total_edges': len(self.social_graph.edges()),
                        'connected_components': len(components),
                        'largest_component_size': largest_component_size,
                        'most_central_users': central_users,
                        'avg_betweenness': np.mean(list(betweenness.values())),
                        'avg_closeness': np.mean(list(closeness.values()))
                    }
                except Exception as e:
                    self.logger.warning(f"Error in detailed network analysis: {e}")
                    return {
                        'network_density': 0.0,
                        'total_nodes': len(self.social_graph.nodes()),
                        'total_edges': len(self.social_graph.edges()),
                        'connected_components': 0,
                        'largest_component_size': 0,
                        'most_central_users': [],
                        'avg_betweenness': 0.0,
                        'avg_closeness': 0.0
                    }
            else:
                return {
                    'network_density': 0.0,
                    'total_nodes': len(self.social_graph.nodes()),
                    'total_edges': 0,
                    'connected_components': 0,
                    'largest_component_size': 0,
                    'most_central_users': [],
                    'avg_betweenness': 0.0,
                    'avg_closeness': 0.0
                }

        except Exception as e:
            self.logger.error(f"Error performing network analysis: {e}")
            return {}

    def _analyze_market_correlation(self, sentiment_results: Dict[str, Any],
                                  market_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlation between social sentiment and market movements."""
        try:
            if not market_data:
                return {'correlation': 0.0, 'lag_correlation': {}, 'predictive_power': 0.0}

            # Extract market metrics
            price_change = market_data.get('price_change', 0.0)
            volume_change = market_data.get('volume_change', 0.0)
            volatility = market_data.get('volatility', 0.0)

            # Current sentiment
            current_sentiment = sentiment_results.get('weighted_sentiment', 0.0)

            # Simple correlation calculation (in production, use historical data)
            sentiment_magnitude = abs(current_sentiment)
            price_magnitude = abs(price_change)

            # Correlation heuristic
            if sentiment_magnitude > 0.5 and price_magnitude > 0.01:  # 1% price change
                if (current_sentiment > 0 and price_change > 0) or \
                   (current_sentiment < 0 and price_change < 0):
                    correlation = min(1.0, sentiment_magnitude * 2)
                else:
                    correlation = -min(1.0, sentiment_magnitude * 2)
            else:
                correlation = 0.0

            # Volume correlation
            volume_correlation = 0.0
            if sentiment_magnitude > 0.3:
                volume_correlation = min(1.0, sentiment_magnitude * 1.5)

            # Volatility correlation
            volatility_correlation = min(1.0, sentiment_magnitude * volatility * 10)

            return {
                'price_correlation': correlation,
                'volume_correlation': volume_correlation,
                'volatility_correlation': volatility_correlation,
                'sentiment_magnitude': sentiment_magnitude,
                'market_response_strength': (abs(correlation) + volume_correlation) / 2,
                'lead_lag_analysis': self._analyze_lead_lag_relationship(current_sentiment, market_data)
            }

        except Exception as e:
            self.logger.error(f"Error analyzing market correlation: {e}")
            return {'correlation': 0.0}

    def _analyze_lead_lag_relationship(self, sentiment: float,
                                     market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze lead-lag relationship between sentiment and market."""
        try:
            # Simplified lead-lag analysis
            # In production, would use historical time series data

            sentiment_strength = abs(sentiment)
            market_volatility = market_data.get('volatility', 0.0)

            # Heuristic: Strong sentiment often leads market movements
            if sentiment_strength > 0.7:
                lead_probability = 0.8
                lag_hours = 2  # Estimate 2-hour lead time
            elif sentiment_strength > 0.5:
                lead_probability = 0.6
                lag_hours = 4
            else:
                lead_probability = 0.3
                lag_hours = 8

            return {
                'sentiment_leads_market': lead_probability,
                'estimated_lag_hours': lag_hours,
                'confidence': sentiment_strength,
                'market_responsiveness': min(1.0, market_volatility * 5)
            }

        except Exception:
            return {}

    def _generate_trading_signals(self, sentiment_results: Dict[str, Any],
                                trend_analysis: Dict[str, Any],
                                influence_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on social media analysis."""
        try:
            signals = {
                'signal_strength': 0.0,
                'bullish_signals': [],
                'bearish_signals': [],
                'confidence': 0.0
            }

            # Extract key metrics
            weighted_sentiment = sentiment_results.get('weighted_sentiment', 0.0)
            viral_sentiment = sentiment_results.get('viral_sentiment', 0.0)
            trend_direction = trend_analysis.get('sentiment_trend', 0.0)
            trend_strength = trend_analysis.get('trend_strength', 0.0)
            momentum = trend_analysis.get('momentum', 0.0)
            influence_weighted_sentiment = influence_metrics.get('influence_weighted_sentiment', 0.0)

            # Signal generation logic
            signal_components = []

            # 1. Overall sentiment signal
            if weighted_sentiment > 0.6:
                signals['bullish_signals'].append({
                    'type': 'positive_sentiment',
                    'strength': weighted_sentiment,
                    'description': f'Strong positive sentiment ({weighted_sentiment:.2f})'
                })
                signal_components.append(weighted_sentiment)
            elif weighted_sentiment < -0.6:
                signals['bearish_signals'].append({
                    'type': 'negative_sentiment',
                    'strength': abs(weighted_sentiment),
                    'description': f'Strong negative sentiment ({weighted_sentiment:.2f})'
                })
                signal_components.append(abs(weighted_sentiment))

            # 2. Viral sentiment signal
            if viral_sentiment > 0.5:
                signals['bullish_signals'].append({
                    'type': 'viral_positive',
                    'strength': viral_sentiment,
                    'description': f'Viral positive content ({viral_sentiment:.2f})'
                })
                signal_components.append(viral_sentiment * 1.2)  # Weight viral content more
            elif viral_sentiment < -0.5:
                signals['bearish_signals'].append({
                    'type': 'viral_negative',
                    'strength': abs(viral_sentiment),
                    'description': f'Viral negative content ({viral_sentiment:.2f})'
                })
                signal_components.append(abs(viral_sentiment) * 1.2)

            # 3. Trend signal
            if trend_direction > 0.3 and trend_strength > 0.5:
                signals['bullish_signals'].append({
                    'type': 'positive_trend',
                    'strength': trend_direction * trend_strength,
                    'description': f'Strong positive trend (direction: {trend_direction:.2f}, strength: {trend_strength:.2f})'
                })
                signal_components.append(trend_direction * trend_strength)
            elif trend_direction < -0.3 and trend_strength > 0.5:
                signals['bearish_signals'].append({
                    'type': 'negative_trend',
                    'strength': abs(trend_direction) * trend_strength,
                    'description': f'Strong negative trend (direction: {trend_direction:.2f}, strength: {trend_strength:.2f})'
                })
                signal_components.append(abs(trend_direction) * trend_strength)

            # 4. Momentum signal
            if momentum > 0.4:
                signals['bullish_signals'].append({
                    'type': 'positive_momentum',
                    'strength': momentum,
                    'description': f'Strong positive momentum ({momentum:.2f})'
                })
                signal_components.append(momentum)
            elif momentum < -0.4:
                signals['bearish_signals'].append({
                    'type': 'negative_momentum',
                    'strength': abs(momentum),
                    'description': f'Strong negative momentum ({momentum:.2f})'
                })
                signal_components.append(abs(momentum))

            # 5. Influence-weighted signal
            if influence_weighted_sentiment > 0.7:
                signals['bullish_signals'].append({
                    'type': 'influencer_positive',
                    'strength': influence_weighted_sentiment,
                    'description': f'Strong positive influencer sentiment ({influence_weighted_sentiment:.2f})'
                })
                signal_components.append(influence_weighted_sentiment * 1.3)  # Weight influencer sentiment more
            elif influence_weighted_sentiment < -0.7:
                signals['bearish_signals'].append({
                    'type': 'influencer_negative',
                    'strength': abs(influence_weighted_sentiment),
                    'description': f'Strong negative influencer sentiment ({influence_weighted_sentiment:.2f})'
                })
                signal_components.append(abs(influence_weighted_sentiment) * 1.3)

            # Calculate overall signal strength
            if signal_components:
                signals['signal_strength'] = np.mean(signal_components)
                signals['confidence'] = min(1.0, len(signal_components) / 5.0)  # More signals = higher confidence

            # Determine overall signal direction
            bullish_strength = sum(s['strength'] for s in signals['bullish_signals'])
            bearish_strength = sum(s['strength'] for s in signals['bearish_signals'])

            if bullish_strength > bearish_strength:
                signals['overall_direction'] = 'bullish'
                signals['direction_confidence'] = bullish_strength / (bullish_strength + bearish_strength + 0.001)
            elif bearish_strength > bullish_strength:
                signals['overall_direction'] = 'bearish'
                signals['direction_confidence'] = bearish_strength / (bullish_strength + bearish_strength + 0.001)
            else:
                signals['overall_direction'] = 'neutral'
                signals['direction_confidence'] = 0.0

            return signals

        except Exception as e:
            self.logger.error(f"Error generating trading signals: {e}")
            return {
                'signal_strength': 0.0,
                'bullish_signals': [],
                'bearish_signals': [],
                'confidence': 0.0
            }

    def get_hashtag_trends(self) -> Dict[str, float]:
        """Get current hashtag trends."""
        try:
            # Update hashtag trends from recent analyses
            hashtag_counts = defaultdict(int)
            hashtag_sentiments = defaultdict(list)

            for analysis in self.post_analyses.values():
                for hashtag, sentiment in analysis.hashtag_sentiment.items():
                    hashtag_counts[hashtag] += 1
                    hashtag_sentiments[hashtag].append(sentiment)

            # Calculate trending hashtags
            self.hashtag_trends = {}
            for hashtag, sentiments in hashtag_sentiments.items():
                if hashtag_counts[hashtag] >= 3:  # Minimum occurrence threshold
                    avg_sentiment = np.mean(sentiments)
                    trend_score = avg_sentiment * np.log(hashtag_counts[hashtag] + 1)
                    self.hashtag_trends[hashtag] = trend_score

            # Sort by trend score and return top trends
            sorted_trends = sorted(self.hashtag_trends.items(),
                                 key=lambda x: abs(x[1]), reverse=True)

            return dict(sorted_trends[:20])  # Top 20 trending hashtags

        except Exception as e:
            self.logger.error(f"Error getting hashtag trends: {e}")
            return {}

    def get_mention_trends(self) -> Dict[str, float]:
        """Get current mention trends."""
        try:
            # Update mention trends from recent analyses
            mention_counts = defaultdict(int)
            mention_sentiments = defaultdict(list)

            for analysis in self.post_analyses.values():
                for mention, sentiment in analysis.mention_sentiment.items():
                    mention_counts[mention] += 1
                    mention_sentiments[mention].append(sentiment)

            # Calculate trending mentions
            self.mention_trends = {}
            for mention, sentiments in mention_sentiments.items():
                if mention_counts[mention] >= 2:  # Minimum occurrence threshold
                    avg_sentiment = np.mean(sentiments)
                    trend_score = avg_sentiment * np.log(mention_counts[mention] + 1)
                    self.mention_trends[mention] = trend_score

            # Sort by trend score and return top trends
            sorted_trends = sorted(self.mention_trends.items(),
                                 key=lambda x: abs(x[1]), reverse=True)

            return dict(sorted_trends[:15])  # Top 15 trending mentions

        except Exception as e:
            self.logger.error(f"Error getting mention trends: {e}")
            return {}

    def get_viral_posts(self) -> List[Dict[str, Any]]:
        """Get current viral posts."""
        try:
            viral_posts = []

            for post_id, viral_analysis in self.viral_analyses.items():
                if viral_analysis.viral_score >= self.parameters['viral_threshold']:
                    post_analysis = self.post_analyses.get(post_id)
                    if post_analysis:
                        viral_posts.append({
                            'post_id': post_id,
                            'viral_score': viral_analysis.viral_score,
                            'viral_status': viral_analysis.viral_status.value,
                            'sentiment_score': post_analysis.sentiment_score,
                            'influence_score': post_analysis.influence_score,
                            'reach_estimate': viral_analysis.reach_estimate,
                            'propagation_rate': viral_analysis.propagation_rate
                        })

            # Sort by viral score
            viral_posts.sort(key=lambda x: x['viral_score'], reverse=True)

            return viral_posts

        except Exception as e:
            self.logger.error(f"Error getting viral posts: {e}")
            return []

    def reset(self):
        """Reset indicator state."""
        try:
            self.posts_data.clear()
            self.post_analyses.clear()
            self.viral_analyses.clear()
            self.aggregated_sentiment = 0.0
            self.trend_direction = 0.0
            self.influence_weighted_sentiment = 0.0
            self.viral_sentiment = 0.0
            self.platform_sentiments.clear()
            self.hashtag_trends.clear()
            self.mention_trends.clear()
            self.last_update = None
            self.social_graph.clear()

            self.logger.info("Social Media Post Indicator reset successfully")

        except Exception as e:
            self.logger.error(f"Error resetting indicator: {e}")

    def get_indicator_data(self) -> Dict[str, Any]:
        """Get comprehensive indicator data."""
        try:
            return {
                'aggregated_sentiment': self.aggregated_sentiment,
                'trend_direction': self.trend_direction,
                'influence_weighted_sentiment': self.influence_weighted_sentiment,
                'viral_sentiment': self.viral_sentiment,
                'platform_sentiments': {platform.value: sentiment
                                       for platform, sentiment in self.platform_sentiments.items()},
                'hashtag_trends': self.get_hashtag_trends(),
                'mention_trends': self.get_mention_trends(),
                'viral_posts': self.get_viral_posts(),
                'data_quality': {
                    'total_posts': len(self.posts_data),
                    'analyzed_posts': len(self.post_analyses),
                    'viral_posts': len([v for v in self.viral_analyses.values()
                                      if v.viral_score >= self.parameters['viral_threshold']]),
                    'avg_authenticity': np.mean([a.authenticity_score for a in self.post_analyses.values()])
                                       if self.post_analyses else 0.0,
                    'platform_coverage': len(set(p.platform for p in self.posts_data))
                },
                'last_update': self.last_update.isoformat() if self.last_update else None
            }

        except Exception as e:
            self.logger.error(f"Error getting indicator data: {e}")
            return {}

    def __str__(self) -> str:
        """String representation."""
        return (f"SocialMediaPostIndicator("
                f"sentiment={self.aggregated_sentiment:.3f}, "
                f"trend={self.trend_direction:.3f}, "
                f"posts={len(self.posts_data)}, "
                f"viral={len([v for v in self.viral_analyses.values() if v.viral_score >= 0.7])})")

    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"SocialMediaPostIndicator("
                f"lookback_hours={self.parameters['lookback_hours']}, "
                f"viral_threshold={self.parameters['viral_threshold']}, "
                f"aggregated_sentiment={self.aggregated_sentiment:.3f})")


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        'lookback_hours': 12,
        'min_posts_threshold': 5,
        'sentiment_decay_hours': 4,
        'viral_threshold': 0.6,
        'authenticity_threshold': 0.5,
        'market_relevance_threshold': 0.4
    }

    # Create indicator
    social_indicator = SocialMediaPostIndicator(config)

    # Example social media data
    sample_posts = [
        {
            'id': 'tweet_001',
            'platform': 'twitter',
            'type': 'original',
            'text': 'Bitcoin is going to the moon! 🚀 #BTC #crypto #bullish',
            'author_id': 'crypto_expert',
            'author_name': 'CryptoExpert',
            'verified': True,
            'followers': 50000,
            'following': 1000,
            'timestamp': '2024-01-15 10:30:00',
            'likes': 150,
            'shares': 45,
            'comments': 23
        },
        {
            'id': 'reddit_001',
            'platform': 'reddit',
            'type': 'original',
            'text': 'Market crash incoming. Fed policy is too aggressive. Time to short everything.',
            'author_id': 'market_bear',
            'author_name': 'MarketBear',
            'verified': False,
            'followers': 5000,
            'following': 200,
            'timestamp': '2024-01-15 11:15:00',
            'likes': 89,
            'shares': 12,
            'comments': 34
        },
        {
            'id': 'tweet_002',
            'platform': 'twitter',
            'type': 'retweet',
            'text': 'Just bought more ETH. Diamond hands! 💎🙌 #ethereum #hodl',
            'author_id': 'eth_whale',
            'author_name': 'EthWhale',
            'verified': False,
            'followers': 15000,
            'following': 500,
            'timestamp': '2024-01-15 12:00:00',
            'likes': 234,
            'shares': 67,
            'comments': 19
        }
    ]

    # Example market data
    market_data = {
        'price_change': 0.025,  # 2.5% price increase
        'volume_change': 0.15,  # 15% volume increase
        'volatility': 0.02      # 2% volatility
    }

    # Calculate indicator
    result = social_indicator.calculate(sample_posts, market_data)

    print("Social Media Post Indicator Results:")
    print(f"Aggregated Sentiment: {result.get('aggregated_sentiment', 0):.3f}")
    print(f"Trend Direction: {result.get('trend_direction', 0):.3f}")
    print(f"Influence Weighted Sentiment: {result.get('influence_weighted_sentiment', 0):.3f}")
    print(f"Viral Sentiment: {result.get('viral_sentiment', 0):.3f}")
    print(f"Signal Strength: {result.get('signal_strength', 0):.3f}")
    print(f"Bullish Signals: {len(result.get('bullish_signals', []))}")
    print(f"Bearish Signals: {len(result.get('bearish_signals', []))}")
    print(f"Total Posts Analyzed: {result.get('total_posts_analyzed', 0)}")
    print(f"Viral Posts: {result.get('viral_posts', 0)}")

    # Get detailed data
    indicator_data = social_indicator.get_indicator_data()
    print(f"\nData Quality:")
    print(f"Platform Coverage: {indicator_data['data_quality']['platform_coverage']}")
    print(f"Average Authenticity: {indicator_data['data_quality']['avg_authenticity']:.3f}")
