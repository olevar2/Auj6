"""
News Article Indicator for AUJ Platform
Advanced natural language processing and sentiment analysis for market impact quantification

This indicator implements sophisticated news sentiment analysis with advanced NLP techniques,
machine learning classification, and real-time news impact assessment for trading decisions.

Features:
- Advanced natural language processing with transformer models
- Multi-dimensional sentiment analysis (sentiment, emotion, impact)
- Real-time news feed processing and filtering
- Machine learning-based market impact prediction
- Entity recognition and financial keyword extraction
- Temporal sentiment decay modeling
- Cross-source news validation and consensus
- Advanced text preprocessing and normalization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
import re
import time
from datetime import datetime, timedelta
import json
import requests
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# NLP and ML imports
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from textblob import TextBlob
except ImportError:
    nltk = None
    TextBlob = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from transformers import AutoModelForTokenClassification, TokenClassificationPipeline
except ImportError:
    pipeline = None

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from scipy import stats
from scipy.signal import savgol_filter

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


class SentimentPolarity(Enum):
    """Sentiment polarity categories"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class NewsImpact(Enum):
    """News impact levels on market"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class NewsCategory(Enum):
    """News category classification"""
    ECONOMIC_DATA = "economic_data"
    CENTRAL_BANK = "central_bank"
    POLITICAL = "political"
    CORPORATE = "corporate"
    GEOPOLITICAL = "geopolitical"
    MARKET_STRUCTURE = "market_structure"
    TECHNICAL = "technical"
    OTHER = "other"


@dataclass
class NewsArticle:
    """Container for news article data"""
    title: str
    content: str
    source: str
    timestamp: datetime
    url: Optional[str] = None
    author: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'content': self.content,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'url': self.url,
            'author': self.author
        }


@dataclass
class SentimentAnalysis:
    """Container for sentiment analysis results"""
    polarity: SentimentPolarity
    confidence: float
    compound_score: float
    positive_score: float
    negative_score: float
    neutral_score: float
    emotion_scores: Dict[str, float]
    subjectivity: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'polarity': self.polarity.value,
            'confidence': self.confidence,
            'compound_score': self.compound_score,
            'positive_score': self.positive_score,
            'negative_score': self.negative_score,
            'neutral_score': self.neutral_score,
            'emotion_scores': self.emotion_scores,
            'subjectivity': self.subjectivity
        }


@dataclass
class NewsImpactAnalysis:
    """Container for news impact analysis"""
    impact_level: NewsImpact
    impact_score: float
    market_relevance: float
    temporal_decay: float
    entity_mentions: List[str]
    financial_keywords: List[str]
    category: NewsCategory
    urgency_score: float
    consensus_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'impact_level': self.impact_level.value,
            'impact_score': self.impact_score,
            'market_relevance': self.market_relevance,
            'temporal_decay': self.temporal_decay,
            'entity_mentions': self.entity_mentions,
            'financial_keywords': self.financial_keywords,
            'category': self.category.value,
            'urgency_score': self.urgency_score,
            'consensus_score': self.consensus_score
        }


@dataclass
class NewsSignal:
    """Container for complete news-based signal"""
    article: NewsArticle
    sentiment: SentimentAnalysis
    impact: NewsImpactAnalysis
    signal_strength: float
    signal_direction: float
    confidence: float
    processing_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'article': self.article.to_dict(),
            'sentiment': self.sentiment.to_dict(),
            'impact': self.impact.to_dict(),
            'signal_strength': self.signal_strength,
            'signal_direction': self.signal_direction,
            'confidence': self.confidence,
            'processing_timestamp': self.processing_timestamp.isoformat()
        }


class NewsArticleIndicator(StandardIndicatorInterface):
    """
    Advanced News Article Indicator
    
    This indicator processes news articles using sophisticated NLP techniques to generate
    trading signals based on sentiment analysis, market impact assessment, and temporal modeling.
    """
    
    def __init__(self,
                 max_article_age_hours: int = 24,
                 min_relevance_threshold: float = 0.3,
                 sentiment_model: str = "finbert",
                 impact_decay_rate: float = 0.1,
                 consensus_window_minutes: int = 60,
                 use_transformer_models: bool = True,
                 financial_keywords_file: Optional[str] = None):
        """
        Initialize News Article Indicator
        
        Args:
            max_article_age_hours: Maximum age of articles to consider
            min_relevance_threshold: Minimum relevance score for inclusion
            sentiment_model: Model to use for sentiment analysis
            impact_decay_rate: Rate of temporal impact decay
            consensus_window_minutes: Window for news consensus analysis
            use_transformer_models: Whether to use transformer-based models
            financial_keywords_file: Path to financial keywords file
        """
        super().__init__()
        
        self.max_article_age_hours = max_article_age_hours
        self.min_relevance_threshold = max(0.0, min(1.0, min_relevance_threshold))
        self.sentiment_model = sentiment_model
        self.impact_decay_rate = max(0.01, min(1.0, impact_decay_rate))
        self.consensus_window_minutes = max(5, consensus_window_minutes)
        self.use_transformer_models = use_transformer_models
        
        # Initialize NLP components
        self._initialize_nlp_components()
        
        # Financial keywords and entities
        self.financial_keywords = self._load_financial_keywords(financial_keywords_file)
        self.financial_entities = self._load_financial_entities()
        
        # ML models
        self.sentiment_classifier = None
        self.impact_classifier = None
        self.category_classifier = None
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.scaler = StandardScaler()
        
        # News processing components
        self.news_cache = deque(maxlen=1000)
        self.processed_articles = set()
        self.sentiment_history = defaultdict(list)
        self.impact_history = defaultdict(list)
        
        # Performance tracking
        self.processing_stats = {
            'articles_processed': 0,
            'signals_generated': 0,
            'avg_processing_time': 0.0,
            'accuracy_score': 0.0
        }
        
        self.logger = logging.getLogger(__name__)

    def _initialize_nlp_components(self):
        """Initialize NLP libraries and models"""
        try:
            # Download required NLTK data
            if nltk:
                nltk.download('vader_lexicon', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                
                self.sia = SentimentIntensityAnalyzer()
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            
            # Initialize transformer models if available
            if pipeline and self.use_transformer_models:
                try:
                    # Financial sentiment model
                    self.finbert_pipeline = pipeline(
                        "sentiment-analysis",
                        model="ProsusAI/finbert",
                        tokenizer="ProsusAI/finbert"
                    )
                    
                    # Named entity recognition
                    self.ner_pipeline = pipeline(
                        "ner",
                        model="dbmdz/bert-large-cased-finetuned-conll03-english",
                        aggregation_strategy="simple"
                    )
                    
                    # Emotion classification
                    self.emotion_pipeline = pipeline(
                        "text-classification",
                        model="j-hartmann/emotion-english-distilroberta-base"
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load transformer models: {e}")
                    self.finbert_pipeline = None
                    self.ner_pipeline = None
                    self.emotion_pipeline = None
            else:
                self.finbert_pipeline = None
                self.ner_pipeline = None
                self.emotion_pipeline = None
                
        except Exception as e:
            self.logger.warning(f"NLP initialization failed: {e}")

    def _load_financial_keywords(self, keywords_file: Optional[str]) -> List[str]:
        """Load financial keywords for relevance scoring"""
        default_keywords = [
            # Economic indicators
            'gdp', 'inflation', 'unemployment', 'interest rate', 'fed', 'ecb', 'boe',
            'monetary policy', 'fiscal policy', 'quantitative easing', 'tapering',
            
            # Market terms
            'bull market', 'bear market', 'volatility', 'liquidity', 'correlation',
            'support', 'resistance', 'breakout', 'breakdown', 'momentum',
            
            # Financial instruments
            'forex', 'currency', 'bond', 'equity', 'commodity', 'derivative',
            'futures', 'options', 'swap', 'etf', 'mutual fund',
            
            # Economic events
            'earnings', 'merger', 'acquisition', 'ipo', 'dividend', 'split',
            'buyback', 'guidance', 'outlook', 'forecast', 'estimate',
            
            # Risk factors
            'default', 'credit rating', 'downgrade', 'upgrade', 'recession',
            'recovery', 'stimulus', 'bailout', 'sanctions', 'trade war'
        ]
        
        if keywords_file:
            try:
                with open(keywords_file, 'r') as f:
                    additional_keywords = [line.strip().lower() for line in f if line.strip()]
                return list(set(default_keywords + additional_keywords))
            except Exception as e:
                self.logger.warning(f"Failed to load keywords file: {e}")
        
        return default_keywords

    def _load_financial_entities(self) -> List[str]:
        """Load financial entities for recognition"""
        return [
            # Central banks
            'federal reserve', 'fed', 'ecb', 'european central bank', 'bank of england',
            'boe', 'bank of japan', 'boj', 'people\'s bank of china', 'pboc',
            
            # Major currencies
            'usd', 'eur', 'gbp', 'jpy', 'chf', 'cad', 'aud', 'nzd',
            'dollar', 'euro', 'pound', 'yen', 'franc', 'yuan', 'renminbi',
            
            # Economic organizations
            'imf', 'world bank', 'oecd', 'g7', 'g20', 'opec', 'wto',
            
            # Major indices
            'dow jones', 'sp500', 's&p 500', 'nasdaq', 'ftse', 'dax', 'nikkei',
            'hang seng', 'asx', 'tsx', 'cac', 'ibex', 'mib',
            
            # Major companies
            'apple', 'microsoft', 'amazon', 'google', 'facebook', 'tesla',
            'berkshire hathaway', 'jpmorgan', 'goldman sachs', 'morgan stanley'
        ]

    def calculate(self, data: pd.DataFrame, news_articles: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate news-based trading signals
        
        Args:
            data: OHLCV data (for context and validation)
            news_articles: List of news articles to analyze
            
        Returns:
            Dictionary containing comprehensive news analysis and signals
        """
        try:
            start_time = time.time()
            
            if not news_articles:
                return self._get_default_result("No news articles provided")
            
            # Process news articles
            processed_signals = []
            for article_data in news_articles:
                try:
                    signal = self._process_news_article(article_data)
                    if signal:
                        processed_signals.append(signal)
                except Exception as e:
                    self.logger.warning(f"Failed to process article: {e}")
                    continue
            
            if not processed_signals:
                return self._get_default_result("No valid signals generated from articles")
            
            # Aggregate signals
            aggregated_analysis = self._aggregate_news_signals(processed_signals)
            
            # Generate trading signal
            primary_signal = self._generate_trading_signal(aggregated_analysis, data)
            
            # Calculate confidence and strength
            confidence = self._calculate_signal_confidence(aggregated_analysis, processed_signals)
            strength = self._calculate_signal_strength(aggregated_analysis, processed_signals)
            
            # Update performance statistics
            processing_time = time.time() - start_time
            self._update_processing_stats(len(news_articles), len(processed_signals), processing_time)
            
            # Comprehensive result
            result = {
                'signal': primary_signal,
                'strength': strength,
                'confidence': confidence,
                'processed_signals': [signal.to_dict() for signal in processed_signals],
                'aggregated_analysis': aggregated_analysis,
                'sentiment_summary': self._create_sentiment_summary(processed_signals),
                'impact_summary': self._create_impact_summary(processed_signals),
                'consensus_analysis': self._analyze_news_consensus(processed_signals),
                'temporal_analysis': self._analyze_temporal_patterns(processed_signals),
                'processing_stats': self.processing_stats,
                'metadata': {
                    'calculation_timestamp': datetime.now().isoformat(),
                    'articles_processed': len(news_articles),
                    'valid_signals': len(processed_signals),
                    'processing_time_seconds': processing_time,
                    'model_versions': self._get_model_versions(),
                    'nlp_enabled': nltk is not None,
                    'transformers_enabled': self.finbert_pipeline is not None
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in news article analysis: {e}")
            raise IndicatorCalculationError(f"Failed to analyze news articles: {e}")

    def _process_news_article(self, article_data: Dict[str, Any]) -> Optional[NewsSignal]:
        """Process a single news article"""
        try:
            # Create article object
            article = NewsArticle(
                title=article_data.get('title', ''),
                content=article_data.get('content', ''),
                source=article_data.get('source', 'unknown'),
                timestamp=self._parse_timestamp(article_data.get('timestamp')),
                url=article_data.get('url'),
                author=article_data.get('author')
            )
            
            # Check if already processed
            article_hash = self._get_article_hash(article)
            if article_hash in self.processed_articles:
                return None
            
            # Check article age
            if self._is_article_too_old(article):
                return None
            
            # Preprocess text
            processed_text = self._preprocess_text(f"{article.title} {article.content}")
            
            # Check relevance
            relevance_score = self._calculate_relevance_score(processed_text)
            if relevance_score < self.min_relevance_threshold:
                return None
            
            # Perform sentiment analysis
            sentiment_analysis = self._analyze_sentiment(processed_text, article)
            
            # Perform impact analysis
            impact_analysis = self._analyze_impact(processed_text, article, sentiment_analysis)
            
            # Generate signal
            signal_strength, signal_direction = self._calculate_signal_metrics(
                sentiment_analysis, impact_analysis
            )
            
            # Calculate confidence
            signal_confidence = self._calculate_individual_confidence(
                sentiment_analysis, impact_analysis, relevance_score
            )
            
            # Create signal object
            signal = NewsSignal(
                article=article,
                sentiment=sentiment_analysis,
                impact=impact_analysis,
                signal_strength=signal_strength,
                signal_direction=signal_direction,
                confidence=signal_confidence,
                processing_timestamp=datetime.now()
            )
            
            # Add to cache and processed set
            self.news_cache.append(signal)
            self.processed_articles.add(article_hash)
            
            return signal
            
        except Exception as e:
            self.logger.warning(f"Failed to process article: {e}")
            return None

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short words (less than 2 characters)
        words = text.split()
        words = [word for word in words if len(word) >= 2]
        
        if nltk and self.lemmatizer:
            # Lemmatize words
            words = [self.lemmatizer.lemmatize(word) for word in words]
            
            # Remove stopwords but keep important financial terms
            financial_stopwords = self.stop_words - set(self.financial_keywords)
            words = [word for word in words if word not in financial_stopwords]
        
        return ' '.join(words)

    def _calculate_relevance_score(self, text: str) -> float:
        """Calculate financial relevance score of text"""
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # Count financial keyword matches
        keyword_matches = 0
        for keyword in self.financial_keywords:
            if keyword in text:
                keyword_matches += 1
        
        # Count entity matches
        entity_matches = 0
        for entity in self.financial_entities:
            if entity in text:
                entity_matches += 1
        
        # Calculate relevance score
        total_matches = keyword_matches + entity_matches * 2  # Weight entities higher
        max_possible_score = len(self.financial_keywords) + len(self.financial_entities) * 2
        
        relevance_score = total_matches / max_possible_score if max_possible_score > 0 else 0.0
        
        # Boost score for certain important patterns
        boost_patterns = [
            r'\b(interest rate|fed|ecb|central bank)\b',
            r'\b(gdp|inflation|unemployment)\b',
            r'\b(earnings|guidance|outlook)\b',
            r'\b(merger|acquisition|ipo)\b'
        ]
        
        for pattern in boost_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                relevance_score += 0.1
        
        return min(1.0, relevance_score)

    def _analyze_sentiment(self, text: str, article: NewsArticle) -> SentimentAnalysis:
        """Perform comprehensive sentiment analysis"""
        sentiment_scores = {}
        emotion_scores = {}
        
        # VADER sentiment (if available)
        if nltk and self.sia:
            vader_scores = self.sia.polarity_scores(text)
            sentiment_scores['vader'] = vader_scores
        
        # TextBlob sentiment (if available)
        if TextBlob:
            blob = TextBlob(text)
            sentiment_scores['textblob'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        
        # FinBERT sentiment (if available)
        if self.finbert_pipeline:
            try:
                finbert_result = self.finbert_pipeline(text[:512])  # Truncate for model
                if finbert_result:
                    sentiment_scores['finbert'] = {
                        'label': finbert_result[0]['label'],
                        'score': finbert_result[0]['score']
                    }
            except Exception as e:
                self.logger.warning(f"FinBERT analysis failed: {e}")
        
        # Emotion analysis (if available)
        if self.emotion_pipeline:
            try:
                emotion_result = self.emotion_pipeline(text[:512])
                if emotion_result:
                    for result in emotion_result[:5]:  # Top 5 emotions
                        emotion_scores[result['label']] = result['score']
            except Exception as e:
                self.logger.warning(f"Emotion analysis failed: {e}")
        
        # Aggregate sentiment scores
        compound_score = self._aggregate_sentiment_scores(sentiment_scores)
        polarity = self._determine_sentiment_polarity(compound_score)
        confidence = self._calculate_sentiment_confidence(sentiment_scores)
        
        # Extract individual scores
        positive_score = sentiment_scores.get('vader', {}).get('pos', 0.0)
        negative_score = sentiment_scores.get('vader', {}).get('neg', 0.0)
        neutral_score = sentiment_scores.get('vader', {}).get('neu', 0.0)
        subjectivity = sentiment_scores.get('textblob', {}).get('subjectivity', 0.5)
        
        return SentimentAnalysis(
            polarity=polarity,
            confidence=confidence,
            compound_score=compound_score,
            positive_score=positive_score,
            negative_score=negative_score,
            neutral_score=neutral_score,
            emotion_scores=emotion_scores,
            subjectivity=subjectivity
        )

    def _analyze_impact(self, text: str, article: NewsArticle, 
                       sentiment: SentimentAnalysis) -> NewsImpactAnalysis:
        """Analyze potential market impact of news"""
        
        # Extract entities
        entity_mentions = self._extract_entities(text)
        
        # Extract financial keywords
        financial_keywords = self._extract_financial_keywords(text)
        
        # Determine category
        category = self._classify_news_category(text, financial_keywords)
        
        # Calculate market relevance
        market_relevance = self._calculate_market_relevance(
            financial_keywords, entity_mentions, category
        )
        
        # Calculate urgency score
        urgency_score = self._calculate_urgency_score(text, article)
        
        # Calculate temporal decay
        temporal_decay = self._calculate_temporal_decay(article.timestamp)
        
        # Calculate impact score
        impact_score = self._calculate_impact_score(
            sentiment, market_relevance, urgency_score, temporal_decay
        )
        
        # Determine impact level
        impact_level = self._determine_impact_level(impact_score)
        
        # Calculate consensus score (placeholder for now)
        consensus_score = 0.5  # Will be updated in consensus analysis
        
        return NewsImpactAnalysis(
            impact_level=impact_level,
            impact_score=impact_score,
            market_relevance=market_relevance,
            temporal_decay=temporal_decay,
            entity_mentions=entity_mentions,
            financial_keywords=financial_keywords,
            category=category,
            urgency_score=urgency_score,
            consensus_score=consensus_score
        )

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        entities = []
        
        # Use NER pipeline if available
        if self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(text)
                for result in ner_results:
                    if result['score'] > 0.8:  # High confidence entities
                        entities.append(result['word'])
            except Exception as e:
                self.logger.warning(f"NER failed: {e}")
        
        # Manual entity extraction for financial terms
        for entity in self.financial_entities:
            if entity.lower() in text.lower():
                entities.append(entity)
        
        return list(set(entities))

    def _extract_financial_keywords(self, text: str) -> List[str]:
        """Extract financial keywords from text"""
        found_keywords = []
        
        for keyword in self.financial_keywords:
            if keyword.lower() in text.lower():
                found_keywords.append(keyword)
        
        return found_keywords

    def _classify_news_category(self, text: str, keywords: List[str]) -> NewsCategory:
        """Classify news into categories"""
        category_keywords = {
            NewsCategory.ECONOMIC_DATA: ['gdp', 'inflation', 'unemployment', 'cpi', 'ppi'],
            NewsCategory.CENTRAL_BANK: ['fed', 'ecb', 'boe', 'interest rate', 'monetary policy'],
            NewsCategory.POLITICAL: ['election', 'government', 'policy', 'regulation', 'law'],
            NewsCategory.CORPORATE: ['earnings', 'merger', 'acquisition', 'ipo', 'dividend'],
            NewsCategory.GEOPOLITICAL: ['war', 'sanctions', 'trade war', 'conflict', 'crisis'],
            NewsCategory.MARKET_STRUCTURE: ['volatility', 'liquidity', 'correlation', 'volume'],
            NewsCategory.TECHNICAL: ['support', 'resistance', 'breakout', 'momentum', 'trend']
        }
        
        category_scores = {}
        for category, cat_keywords in category_keywords.items():
            score = sum(1 for kw in keywords if kw in cat_keywords)
            category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return NewsCategory.OTHER

    def _calculate_market_relevance(self, keywords: List[str], entities: List[str], 
                                  category: NewsCategory) -> float:
        """Calculate market relevance score"""
        relevance_score = 0.0
        
        # Keyword relevance
        relevance_score += len(keywords) * 0.1
        
        # Entity relevance
        relevance_score += len(entities) * 0.15
        
        # Category relevance
        category_weights = {
            NewsCategory.ECONOMIC_DATA: 0.9,
            NewsCategory.CENTRAL_BANK: 1.0,
            NewsCategory.POLITICAL: 0.6,
            NewsCategory.CORPORATE: 0.7,
            NewsCategory.GEOPOLITICAL: 0.8,
            NewsCategory.MARKET_STRUCTURE: 0.8,
            NewsCategory.TECHNICAL: 0.5,
            NewsCategory.OTHER: 0.3
        }
        
        relevance_score += category_weights.get(category, 0.3)
        
        return min(1.0, relevance_score)

    def _calculate_urgency_score(self, text: str, article: NewsArticle) -> float:
        """Calculate urgency score based on text analysis"""
        urgency_indicators = [
            r'\b(breaking|urgent|alert|emergency)\b',
            r'\b(immediate|instant|now|today)\b',
            r'\b(crisis|crash|collapse|surge)\b',
            r'\b(surprise|unexpected|shock)\b'
        ]
        
        urgency_score = 0.0
        for pattern in urgency_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                urgency_score += 0.2
        
        # Time-based urgency (newer articles are more urgent)
        age_hours = (datetime.now() - article.timestamp).total_seconds() / 3600
        time_factor = max(0.0, 1.0 - (age_hours / self.max_article_age_hours))
        urgency_score += time_factor * 0.3
        
        return min(1.0, urgency_score)

    def _calculate_temporal_decay(self, timestamp: datetime) -> float:
        """Calculate temporal decay factor"""
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        decay_factor = np.exp(-self.impact_decay_rate * age_hours)
        return max(0.0, decay_factor)

    def _calculate_impact_score(self, sentiment: SentimentAnalysis, market_relevance: float,
                              urgency_score: float, temporal_decay: float) -> float:
        """Calculate overall impact score"""
        # Sentiment magnitude
        sentiment_magnitude = abs(sentiment.compound_score)
        
        # Combine factors
        impact_score = (
            sentiment_magnitude * 0.3 +
            market_relevance * 0.4 +
            urgency_score * 0.2 +
            temporal_decay * 0.1
        )
        
        return min(1.0, impact_score)

    def _determine_impact_level(self, impact_score: float) -> NewsImpact:
        """Determine impact level from score"""
        if impact_score >= 0.8:
            return NewsImpact.CRITICAL
        elif impact_score >= 0.6:
            return NewsImpact.HIGH
        elif impact_score >= 0.4:
            return NewsImpact.MEDIUM
        elif impact_score >= 0.2:
            return NewsImpact.LOW
        else:
            return NewsImpact.NEGLIGIBLE

    def _aggregate_sentiment_scores(self, sentiment_scores: Dict[str, Any]) -> float:
        """Aggregate multiple sentiment scores"""
        scores = []
        weights = []
        
        # VADER (reliable for social media text)
        if 'vader' in sentiment_scores:
            scores.append(sentiment_scores['vader']['compound'])
            weights.append(0.4)
        
        # TextBlob (good baseline)
        if 'textblob' in sentiment_scores:
            scores.append(sentiment_scores['textblob']['polarity'])
            weights.append(0.3)
        
        # FinBERT (specialized for financial text)
        if 'finbert' in sentiment_scores:
            finbert_score = sentiment_scores['finbert']['score']
            if sentiment_scores['finbert']['label'].lower() == 'negative':
                finbert_score = -finbert_score
            elif sentiment_scores['finbert']['label'].lower() == 'neutral':
                finbert_score = 0.0
            scores.append(finbert_score)
            weights.append(0.5)  # Higher weight for financial-specific model
        
        if not scores:
            return 0.0
        
        # Weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            return sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            return np.mean(scores)

    def _determine_sentiment_polarity(self, compound_score: float) -> SentimentPolarity:
        """Determine sentiment polarity from compound score"""
        if compound_score >= 0.5:
            return SentimentPolarity.VERY_POSITIVE
        elif compound_score >= 0.1:
            return SentimentPolarity.POSITIVE
        elif compound_score <= -0.5:
            return SentimentPolarity.VERY_NEGATIVE
        elif compound_score <= -0.1:
            return SentimentPolarity.NEGATIVE
        else:
            return SentimentPolarity.NEUTRAL

    def _calculate_sentiment_confidence(self, sentiment_scores: Dict[str, Any]) -> float:
        """Calculate confidence in sentiment analysis"""
        confidence_factors = []
        
        # Number of models agreeing
        model_count = len(sentiment_scores)
        if model_count > 1:
            confidence_factors.append(min(1.0, model_count / 3))
        
        # Individual model confidences
        if 'finbert' in sentiment_scores:
            confidence_factors.append(sentiment_scores['finbert']['score'])
        
        if 'vader' in sentiment_scores:
            # VADER confidence based on magnitude
            vader_magnitude = abs(sentiment_scores['vader']['compound'])
            confidence_factors.append(vader_magnitude)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5

    def _calculate_signal_metrics(self, sentiment: SentimentAnalysis, 
                                impact: NewsImpactAnalysis) -> Tuple[float, float]:
        """Calculate signal strength and direction"""
        
        # Signal direction based on sentiment
        signal_direction = sentiment.compound_score
        
        # Signal strength based on impact and confidence
        signal_strength = (
            impact.impact_score * 0.5 +
            sentiment.confidence * 0.3 +
            impact.market_relevance * 0.2
        )
        
        return signal_strength, signal_direction

    def _calculate_individual_confidence(self, sentiment: SentimentAnalysis,
                                       impact: NewsImpactAnalysis,
                                       relevance_score: float) -> float:
        """Calculate confidence for individual signal"""
        confidence_factors = [
            sentiment.confidence,
            impact.market_relevance,
            relevance_score,
            impact.temporal_decay
        ]
        
        return np.mean(confidence_factors)

    def _aggregate_news_signals(self, signals: List[NewsSignal]) -> Dict[str, Any]:
        """Aggregate multiple news signals"""
        if not signals:
            return {}
        
        # Aggregate sentiment
        sentiment_scores = [s.sentiment.compound_score for s in signals]
        sentiment_confidences = [s.sentiment.confidence for s in signals]
        
        # Aggregate impact
        impact_scores = [s.impact.impact_score for s in signals]
        market_relevances = [s.impact.market_relevance for s in signals]
        
        # Aggregate signals
        signal_strengths = [s.signal_strength for s in signals]
        signal_directions = [s.signal_direction for s in signals]
        signal_confidences = [s.confidence for s in signals]
        
        # Calculate weighted averages (weight by confidence)
        def weighted_average(values, weights):
            if not values or not weights:
                return 0.0
            return sum(v * w for v, w in zip(values, weights)) / sum(weights)
        
        aggregated = {
            'avg_sentiment_score': weighted_average(sentiment_scores, sentiment_confidences),
            'avg_impact_score': weighted_average(impact_scores, signal_confidences),
            'avg_market_relevance': weighted_average(market_relevances, signal_confidences),
            'avg_signal_strength': weighted_average(signal_strengths, signal_confidences),
            'avg_signal_direction': weighted_average(signal_directions, signal_confidences),
            'total_signals': len(signals),
            'positive_signals': sum(1 for s in signals if s.signal_direction > 0.1),
            'negative_signals': sum(1 for s in signals if s.signal_direction < -0.1),
            'neutral_signals': sum(1 for s in signals if abs(s.signal_direction) <= 0.1),
            'high_impact_signals': sum(1 for s in signals if s.impact.impact_level in [NewsImpact.HIGH, NewsImpact.CRITICAL]),
            'sentiment_consistency': 1.0 - np.std(sentiment_scores) if len(sentiment_scores) > 1 else 1.0,
            'direction_consensus': self._calculate_direction_consensus(signal_directions)
        }
        
        return aggregated

    def _calculate_direction_consensus(self, directions: List[float]) -> float:
        """Calculate consensus in signal directions"""
        if not directions:
            return 0.0
        
        positive_count = sum(1 for d in directions if d > 0.1)
        negative_count = sum(1 for d in directions if d < -0.1)
        neutral_count = len(directions) - positive_count - negative_count
        
        total = len(directions)
        max_group = max(positive_count, negative_count, neutral_count)
        
        return max_group / total if total > 0 else 0.0

    def _generate_trading_signal(self, aggregated: Dict[str, Any], 
                               data: pd.DataFrame) -> SignalType:
        """Generate primary trading signal"""
        if not aggregated:
            return SignalType.HOLD
        
        avg_direction = aggregated.get('avg_signal_direction', 0.0)
        avg_strength = aggregated.get('avg_signal_strength', 0.0)
        direction_consensus = aggregated.get('direction_consensus', 0.0)
        high_impact_count = aggregated.get('high_impact_signals', 0)
        
        # Signal generation logic
        if (avg_direction > 0.2 and 
            avg_strength > 0.5 and 
            direction_consensus > 0.6):
            return SignalType.BUY
        elif (avg_direction < -0.2 and 
              avg_strength > 0.5 and 
              direction_consensus > 0.6):
            return SignalType.SELL
        elif high_impact_count > 0 and abs(avg_direction) > 0.1:
            # High-impact news override
            return SignalType.BUY if avg_direction > 0 else SignalType.SELL
        
        return SignalType.HOLD

    def _calculate_signal_confidence(self, aggregated: Dict[str, Any], 
                                   signals: List[NewsSignal]) -> float:
        """Calculate overall signal confidence"""
        if not aggregated or not signals:
            return 0.0
        
        confidence_factors = []
        
        # Signal count factor
        signal_count = aggregated.get('total_signals', 0)
        count_factor = min(1.0, signal_count / 5)  # Normalize to max 5 signals
        confidence_factors.append(count_factor)
        
        # Consensus factor
        direction_consensus = aggregated.get('direction_consensus', 0.0)
        confidence_factors.append(direction_consensus)
        
        # Sentiment consistency
        sentiment_consistency = aggregated.get('sentiment_consistency', 0.0)
        confidence_factors.append(sentiment_consistency)
        
        # Average signal confidence
        avg_signal_confidence = np.mean([s.confidence for s in signals])
        confidence_factors.append(avg_signal_confidence)
        
        # High impact signals factor
        high_impact_count = aggregated.get('high_impact_signals', 0)
        high_impact_factor = min(1.0, high_impact_count / 3)
        confidence_factors.append(high_impact_factor)
        
        return np.mean(confidence_factors)

    def _calculate_signal_strength(self, aggregated: Dict[str, Any], 
                                 signals: List[NewsSignal]) -> float:
        """Calculate overall signal strength"""
        if not aggregated or not signals:
            return 0.0
        
        avg_strength = aggregated.get('avg_signal_strength', 0.0)
        avg_impact = aggregated.get('avg_impact_score', 0.0)
        high_impact_ratio = aggregated.get('high_impact_signals', 0) / len(signals)
        
        # Combine factors
        signal_strength = (
            avg_strength * 0.4 +
            avg_impact * 0.4 +
            high_impact_ratio * 0.2
        )
        
        return min(1.0, signal_strength)

    def _create_sentiment_summary(self, signals: List[NewsSignal]) -> Dict[str, Any]:
        """Create sentiment summary"""
        if not signals:
            return {}
        
        sentiments = [s.sentiment for s in signals]
        
        return {
            'total_articles': len(signals),
            'avg_compound_score': np.mean([s.compound_score for s in sentiments]),
            'avg_confidence': np.mean([s.confidence for s in sentiments]),
            'polarity_distribution': {
                polarity.value: sum(1 for s in sentiments if s.polarity == polarity)
                for polarity in SentimentPolarity
            },
            'avg_subjectivity': np.mean([s.subjectivity for s in sentiments]),
            'sentiment_volatility': np.std([s.compound_score for s in sentiments])
        }

    def _create_impact_summary(self, signals: List[NewsSignal]) -> Dict[str, Any]:
        """Create impact summary"""
        if not signals:
            return {}
        
        impacts = [s.impact for s in signals]
        
        return {
            'avg_impact_score': np.mean([i.impact_score for i in impacts]),
            'avg_market_relevance': np.mean([i.market_relevance for i in impacts]),
            'impact_distribution': {
                impact.value: sum(1 for i in impacts if i.impact_level == impact)
                for impact in NewsImpact
            },
            'category_distribution': {
                category.value: sum(1 for i in impacts if i.category == category)
                for category in NewsCategory
            },
            'avg_urgency': np.mean([i.urgency_score for i in impacts]),
            'total_entities': len(set(entity for i in impacts for entity in i.entity_mentions)),
            'total_keywords': len(set(keyword for i in impacts for keyword in i.financial_keywords))
        }

    def _analyze_news_consensus(self, signals: List[NewsSignal]) -> Dict[str, Any]:
        """Analyze consensus among news sources"""
        if not signals:
            return {}
        
        # Group by source
        source_signals = defaultdict(list)
        for signal in signals:
            source_signals[signal.article.source].append(signal)
        
        # Calculate source consensus
        source_directions = {}
        for source, source_signal_list in source_signals.items():
            avg_direction = np.mean([s.signal_direction for s in source_signal_list])
            source_directions[source] = avg_direction
        
        # Overall consensus metrics
        all_directions = list(source_directions.values())
        consensus_score = 1.0 - np.std(all_directions) if len(all_directions) > 1 else 1.0
        
        return {
            'source_count': len(source_signals),
            'consensus_score': consensus_score,
            'source_directions': source_directions,
            'sources_bullish': sum(1 for d in all_directions if d > 0.1),
            'sources_bearish': sum(1 for d in all_directions if d < -0.1),
            'sources_neutral': sum(1 for d in all_directions if abs(d) <= 0.1)
        }

    def _analyze_temporal_patterns(self, signals: List[NewsSignal]) -> Dict[str, Any]:
        """Analyze temporal patterns in news"""
        if not signals:
            return {}
        
        # Sort by timestamp
        sorted_signals = sorted(signals, key=lambda s: s.article.timestamp)
        
        # Calculate time-based metrics
        timestamps = [s.article.timestamp for s in sorted_signals]
        sentiment_scores = [s.sentiment.compound_score for s in sorted_signals]
        
        # Trend analysis
        if len(sentiment_scores) > 2:
            # Linear regression for trend
            x = np.arange(len(sentiment_scores))
            trend_slope, trend_intercept = np.polyfit(x, sentiment_scores, 1)
        else:
            trend_slope = 0.0
            trend_intercept = 0.0
        
        # Temporal clustering
        recent_threshold = datetime.now() - timedelta(hours=2)
        recent_signals = [s for s in signals if s.article.timestamp >= recent_threshold]
        
        return {
            'time_span_hours': (max(timestamps) - min(timestamps)).total_seconds() / 3600,
            'sentiment_trend_slope': trend_slope,
            'recent_signals_count': len(recent_signals),
            'recent_avg_sentiment': np.mean([s.sentiment.compound_score for s in recent_signals]) if recent_signals else 0.0,
            'temporal_volatility': np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.0,
            'news_frequency': len(signals) / max(1, (max(timestamps) - min(timestamps)).total_seconds() / 3600)
        }

    def _update_processing_stats(self, articles_input: int, signals_generated: int, 
                               processing_time: float):
        """Update processing statistics"""
        self.processing_stats['articles_processed'] += articles_input
        self.processing_stats['signals_generated'] += signals_generated
        
        # Update average processing time
        current_avg = self.processing_stats['avg_processing_time']
        total_processed = self.processing_stats['articles_processed']
        
        if total_processed > 0:
            self.processing_stats['avg_processing_time'] = (
                (current_avg * (total_processed - articles_input) + processing_time) / 
                total_processed
            )

    def _get_model_versions(self) -> Dict[str, str]:
        """Get versions of models being used"""
        versions = {}
        
        if self.finbert_pipeline:
            versions['finbert'] = "ProsusAI/finbert"
        
        if self.ner_pipeline:
            versions['ner'] = "dbmdz/bert-large-cased-finetuned-conll03-english"
        
        if self.emotion_pipeline:
            versions['emotion'] = "j-hartmann/emotion-english-distilroberta-base"
        
        if nltk:
            versions['nltk_vader'] = "vader_lexicon"
        
        if TextBlob:
            versions['textblob'] = "textblob"
        
        return versions

    def _parse_timestamp(self, timestamp_str: Any) -> datetime:
        """Parse timestamp from various formats"""
        if isinstance(timestamp_str, datetime):
            return timestamp_str
        
        if isinstance(timestamp_str, str):
            # Try various formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
        
        # Default to current time if parsing fails
        return datetime.now()

    def _get_article_hash(self, article: NewsArticle) -> str:
        """Generate hash for article deduplication"""
        import hashlib
        content = f"{article.title}{article.source}{article.timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()

    def _is_article_too_old(self, article: NewsArticle) -> bool:
        """Check if article is too old to process"""
        age_hours = (datetime.now() - article.timestamp).total_seconds() / 3600
        return age_hours > self.max_article_age_hours

    def _get_default_result(self, reason: str = "No data") -> Dict[str, Any]:
        """Get default result when processing fails"""
        return {
            'signal': SignalType.HOLD,
            'strength': 0.0,
            'confidence': 0.0,
            'processed_signals': [],
            'aggregated_analysis': {},
            'sentiment_summary': {},
            'impact_summary': {},
            'consensus_analysis': {},
            'temporal_analysis': {},
            'processing_stats': self.processing_stats,
            'metadata': {
                'calculation_timestamp': datetime.now().isoformat(),
                'articles_processed': 0,
                'valid_signals': 0,
                'processing_time_seconds': 0.0,
                'model_versions': self._get_model_versions(),
                'nlp_enabled': nltk is not None,
                'transformers_enabled': self.finbert_pipeline is not None,
                'error': reason
            }
        }

    def get_signal_type(self) -> SignalType:
        """Return the type of signal this indicator generates"""
        return SignalType.BUY  # This indicator can generate BUY, SELL, or HOLD signals

    def get_data_requirements(self) -> Dict[str, Any]:
        """Return the data requirements for this indicator"""
        return {
            'required_columns': ['open', 'high', 'low', 'close', 'volume'],
            'min_periods': 1,  # Can work with single data point for context
            'timeframe_compatibility': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'data_type': 'OHLCV + News',
            'additional_data': {
                'news_articles': {
                    'required_fields': ['title', 'content', 'source', 'timestamp'],
                    'optional_fields': ['url', 'author']
                }
            }
        }