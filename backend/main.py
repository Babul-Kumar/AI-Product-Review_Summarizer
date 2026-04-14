import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import random
import re
import threading
import time
from collections import Counter, OrderedDict, deque
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator

# ==============================================================================
# CONFIGURATION - LOGGER SETUP (before optional dependencies)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# OPTIONAL DEPENDENCIES
# ==============================================================================
try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False
    import json

try:
    import httpx
    USE_HTTPX = True
except ImportError:
    USE_HTTPX = False

try:
    import ahocorasick
    USE_AHOCORASICK = True
except ImportError:
    USE_AHOCORASICK = False
    logger.warning("ahocorasick not installed - using fallback pattern matching")

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    vader_analyzer = SentimentIntensityAnalyzer()
    USE_VADER = True
except ImportError:
    USE_VADER = False

# API Keys
VALID_API_KEYS = set()
raw_keys = os.getenv("API_KEYS", "")
if raw_keys:
    VALID_API_KEYS = {k.strip() for k in raw_keys.split(",") if k.strip()}
if not VALID_API_KEYS:
    logger.warning("No API_KEYS configured - running in open mode")

# Cache settings
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1000"))

# Constants
PLACEHOLDER_API_KEYS = {"your_gemini_api_key_here", "replace_with_real_key"}
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LEGACY_MODEL_ALIASES = {"gemini-pro": DEFAULT_GEMINI_MODEL}

# Performance timeouts
GEMINI_TIMEOUT = 10
REQUEST_TIMEOUT = 60

# Limits
MAX_POINTS = int(os.getenv("MAX_POINTS", "6"))
MAX_NEUTRAL_POINTS = int(os.getenv("MAX_NEUTRAL_POINTS", "2"))
MAX_REVIEWS = int(os.getenv("MAX_REVIEWS", "100"))
MAX_REVIEWS_TO_ANALYZE = int(os.getenv("MAX_REVIEWS_TO_ANALYZE", "30"))
SENTIMENT_POLARITY_THRESHOLD = float(os.getenv("SENTIMENT_POLARITY_THRESHOLD", "0.1"))

# Rate Limiting
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "100"))

# Input Limits
MAX_INPUT_SIZE = 10000
MAX_CLAUSES_FOR_AI = int(os.getenv("MAX_CLAUSES_FOR_AI", "20"))
GEMINI_MAX_CLAUSES = 10
GEMINI_MAX_CHARS = 300

# ⚠️ FIX #1: STRICTER RELEVANCE THRESHOLD (changed from 0.3 to 0.4)
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.4"))
MIN_PRODUCT_INDICATORS = int(os.getenv("MIN_PRODUCT_INDICATORS", "2"))

# API version prefix
API_V1_PREFIX = "/api/v1"
API_V2_PREFIX = "/api/v2"

# Streaming
STREAM_CHUNK_DELAY = 0.02

# Filler/Connector words (frozenset for O(1) lookup)
FILLER_WORDS = frozenset({
    "honestly", "basically", "actually", "literally", "overall",
    "personally", "maybe", "probably", "frankly", "simply",
    "just", "really", "very", "quite", "somewhat", "kinda",
    "sorta", "fairly", "pretty", "rather", "enough", "almost",
})

CONNECTOR_WORDS = frozenset({
    "but", "however", "although", "though", "while", "whereas",
    "yet", "except", "otherwise", "nonetheless", "nevertheless",
    "alternatively", "instead", "also", "plus", "and then",
})

NEGATIONS = frozenset({
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "hardly", "barely", "scarcely", "seldom", "rarely", "without",
    "lack", "lacking", "doesn't", "don't", "didn't", "won't", "wouldn't",
    "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
})

# ==============================================================================
# STRUCTURED WARNINGS MODEL
# ==============================================================================
class WarningDetail(BaseModel):
    type: str
    message: str


# ==============================================================================
# 🚨 OUT-OF-SCOPE EXCEPTION
# ==============================================================================
class OutOfScopeError(Exception):
    """Raised when input is not related to product reviews"""
    def __init__(self, message: str, detected_type: str = "unknown"):
        self.message = message
        self.detected_type = detected_type
        super().__init__(self.message)


# ==============================================================================
# PRODUCT REVIEW RELEVANCE DETECTION
# ==============================================================================

# 🚨 FIX #3: CONTEXT-AWARE IRRELEVANT PATTERNS
# Patterns are designed to only trigger when NOT surrounded by product review context

# Product review indicators (used to cancel irrelevant pattern detection)
PRODUCT_REVIEW_INDICATORS = [
    r"\b(review|reviews|rating|stars|recommend|worth|buy|purchase|product|quality)\b",
    r"\b(pros?|cons?|advantages?|disadvantages?|likes?|dislikes?)\b",
    r"\b(\d+\s*stars?|\d+\/10|rating\s*\d+)\b",
    r"\b(bought|ordered|received|shipping|delivered|arrived|returned|refund)\b",
]

# Compile product indicators
_PRODUCT_INDICATOR_PATTERN = re.compile("|".join(PRODUCT_REVIEW_INDICATORS), re.IGNORECASE)

# Irrelevant patterns - but with context awareness
# Each pattern now has a flag for whether it needs product context to trigger
IRRELEVANT_PATTERNS_CONFIG = [
    # Sports - High confidence when standalone
    {"pattern": r"\b(cricket match|football match|soccer match)\b", "weight": 0.8, "needs_context": False},
    {"pattern": r"\b(world cup|super bowl|nba finals|nfl playoffs)\b", "weight": 0.8, "needs_context": False},
    {"pattern": r"\b(play cricket|watch football|go soccer|tennis match|golf game)\b", "weight": 0.8, "needs_context": False},
    # BUT: "cricket bat review" should NOT trigger sports penalty
    {"pattern": r"\b(cricket|football|soccer|basketball|tennis|golf|baseball|hockey|rugby|olympics)\b(?!.*\b(bat|ball|review|product|rating|buy|purchase|amazon|website|store|shop)\b)", "weight": 0.6, "needs_context": True},
    
    # News/Politics - High confidence
    {"pattern": r"\b(election|trump|biden|politics|government|law|congress|senate|parliament|lawyer|court|trial|lawsuit)\b", "weight": 0.9, "needs_context": False},
    
    # Entertainment
    {"pattern": r"\b(netflix series|spotify playlist|youtube video|movie review|film director|actor performance)\b", "weight": 0.8, "needs_context": False},
    {"pattern": r"\b(movie|film|series|netflix|spotify|youtube|instagram|tiktok|facebook|twitter|social media|influencer)\b(?!.*\b(review|recommend|rating|product|buy|quality|worth)\b)", "weight": 0.5, "needs_context": True},
    
    # Finance/Trading
    {"pattern": r"\b(stock market|trading strategy|investing tips|crypto price|bitcoin value|forex trading)\b", "weight": 0.9, "needs_context": False},
    {"pattern": r"\b(stock|market|trading|investing|crypto|bitcoin|ethereum|forex|shares|portfolio|dividend)\b(?!.*\b(review|product|buy|purchase|worth|quality)\b)", "weight": 0.6, "needs_context": True},
    
    # Tech (but not product reviews)
    {"pattern": r"\b(programming tutorial|coding course|algorithm explanation|software development|startup funding|venture capital|ipo news)\b", "weight": 0.9, "needs_context": False},
    
    # Medical/Health
    {"pattern": r"\b(symptoms|diagnosis|treatment|prescription|surgery|therapy|doctor|hospital|medical|health)\b(?!.*\b(review|product|rating|buy|worth|recommend)\b)", "weight": 0.7, "needs_context": True},
    
    # Recipes/Cooking
    {"pattern": r"\b(recipe|cook|bake|chef|kitchen|ingredient|oven|stove|grill)\b(?!.*\b(review|recommend|buy|product|worth|quality)\b)", "weight": 0.6, "needs_context": True},
    
    # Travel
    {"pattern": r"\b(flight booking|hotel reservation|vacation planning|travel tips|airline review|boarding pass|passport|visa)\b", "weight": 0.8, "needs_context": False},
    {"pattern": r"\b(flight|hotel|vacation|travel|trip|booking|airbnb|airline|airport|destination)\b(?!.*\b(review|recommend|buy|product|worth|quality)\b)", "weight": 0.5, "needs_context": True},
    
    # Academic/Educational
    {"pattern": r"\b(university|college|school|exam|grade|homework|assignment|course|study|research|thesis|professor)\b(?!.*\b(review|product|buy|worth|recommend)\b)", "weight": 0.6, "needs_context": True},
    
    # Other non-product
    {"pattern": r"\b(joke|riddle|puzzle|meme|horoscope|astrology|fortune|tarot)\b", "weight": 0.9, "needs_context": False},
]

# Compile patterns
_IRRELEVANT_PATTERN_CONFIGS = IRRELEVANT_PATTERNS_CONFIG

# Product review indicators for scoring
PRODUCT_INDICATORS_SCORING = [
    r"\b(review|reviews|rating|stars|recommend|not recommend|worth it|not worth|buy|don'?t buy|purchase)\b",
    r"\b(product|item|device|gadget|item|brand|model|version)\b",
    r"\b(quality|price|value|durable|reliable|sturdy|flimsy|cheap|premium)\b",
    r"\b(using|used|use|experience|experienced|tried|testing|tested)\b.*\b(product|it|this|that)\b",
    r"\b(product|it|this|that)\b.*\b(using|used|use|experience|experienced|tried|testing|tested)\b",
    r"\b(\d+\s*stars?|\d+\s*out\s*of\s*\d+|rating\s*\d+)\b",
    r"\b(pros?|cons?|advantages?|disadvantages?|likes?|dislikes?|upside|downside)\b",
]

_PRODUCT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PRODUCT_INDICATORS_SCORING]


def has_product_review_context(text: str) -> bool:
    """
    Check if text has strong product review context.
    This is used to determine if irrelevant pattern matches should be ignored.
    """
    # Check for strong product review indicators
    indicator_matches = _PRODUCT_INDICATOR_PATTERN.findall(text)
    
    # If we have 2+ product indicators, it's likely a product review context
    if len(indicator_matches) >= 2:
        return True
    
    # Check for specific high-confidence product review patterns
    strong_patterns = [
        r"\b(review|rating|recommend)\b.*\b(product|quality|worth|price)\b",
        r"\b(product|quality|worth|price)\b.*\b(review|rating|recommend)\b",
        r"\b(buy|purchase|ordered|received)\b.*\b(product|item|thing)\b",
        r"\b(\d+\s*stars?)\b.*\b(product|quality|worth)\b",
    ]
    
    for pattern in strong_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def calculate_relevance_score(text: str) -> float:
    """
    Calculate how relevant the text is to product reviews.
    Returns a score from 0.0 to 1.0.
    
    FIX #3: Context-aware detection - irrelevant patterns don't trigger 
    when the text has strong product review context.
    
    Score breakdown:
    - Contains product review indicators: +0.3 to +0.5
    - Contains product keywords: +0.1 to +0.2
    - Contains irrelevant topics: -0.5 to -0.8 (reduced or cancelled if context present)
    - Text length bonus: +0.0 to +0.1
    """
    text_lower = text.lower()
    score = 0.0
    
    # FIRST: Check if text has strong product review context
    has_review_context = has_product_review_context(text_lower)
    
    # SECOND: Check for irrelevant patterns with context awareness
    irrelevant_matches = []
    for config in _IRRELEVANT_PATTERN_CONFIGS:
        matches = re.findall(config["pattern"], text_lower)
        if matches:
            # If pattern needs context, only apply if NO review context
            if config["needs_context"] and has_review_context:
                # Reduce penalty significantly
                irrelevant_matches.append((matches, config["weight"] * 0.3))
            else:
                irrelevant_matches.append((matches, config["weight"]))
    
    if irrelevant_matches:
        # Calculate weighted penalty
        total_penalty = 0.0
        for matches, weight in irrelevant_matches:
            total_penalty += min(weight * len(matches), 0.6)
        score -= min(total_penalty, 0.8)  # Cap total penalty
    
    # Check for product review indicators
    product_indicator_count = 0
    for pattern in _PRODUCT_PATTERNS:
        if pattern.search(text_lower):
            product_indicator_count += 1
    
    if product_indicator_count >= 4:
        score += 0.5
    elif product_indicator_count >= 2:
        score += 0.35
    elif product_indicator_count == 1:
        score += 0.2
    
    # Check for product-related keywords
    product_keywords = [
        "bought", "purchase", "ordered", "received", "shipping", "delivery",
        "returned", "refund", "warranty", "package", "arrived", 
        "product", "item", "device", "gadget", "thing",
        "better", "worse", "compared", "versus", "vs",
    ]
    keyword_hits = sum(1 for kw in product_keywords if kw in text_lower)
    if keyword_hits >= 4:
        score += 0.2
    elif keyword_hits >= 2:
        score += 0.1
    
    # Length bonus (longer reviews are more likely to be genuine)
    word_count = len(text.split())
    if word_count >= 50:
        score += 0.1
    elif word_count >= 20:
        score += 0.05
    
    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, score))


def is_product_review_related(text: str) -> Tuple[bool, float, str]:
    """
    Check if the input is related to product reviews.
    
    Returns:
        Tuple of (is_relevant, score, detected_type)
    
    FIX #2: "partial" classification is NO LONGER accepted.
    Only "product_review" passes (score >= 0.4).
    
    detected_type can now only be:
        - "product_review" - clearly a product review (score >= 0.4)
        - "irrelevant" - not a product review (score < 0.4)
    """
    score = calculate_relevance_score(text)
    
    # FIX #2: Only accept if score >= 0.4 (clearly a product review)
    # "partial" classification is completely removed
    if score >= 0.4:
        return True, score, "product_review"
    else:
        return False, score, "irrelevant"


def detect_out_of_scope_category(text: str) -> str:
    """
    Detect what category the out-of-scope input falls into.
    FIX #3: Uses context-aware detection.
    """
    text_lower = text.lower()
    
    # First check for strong product review context
    if has_product_review_context(text_lower):
        return "product_review"  # Shouldn't happen if called correctly
    
    categories = {
        "sports": ["cricket match", "football game", "soccer match", "basketball", "tennis match", 
                   "world cup", "super bowl", "player's performance", "team winning", "score prediction"],
        "politics": ["election", "trump", "biden", "politics", "government", "law", "vote", "party", "congress", "senator"],
        "entertainment": ["movie review", "film", "netflix series", "spotify playlist", "youtube video", 
                          "song", "actor", "director", "album", "concert"],
        "finance": ["stock market", "trading", "crypto", "bitcoin", "invest", "shares", "portfolio", "dividend"],
        "tech_news": ["startup funding", "ipo", "acquisition", "layoffs", "tech industry", "programming tutorial"],
        "health": ["symptoms", "diagnosis", "treatment", "doctor", "hospital", "medical", "prescription", "health tips"],
        "travel": ["flight booking", "hotel", "vacation", "booking", "airport", "destination", "itinerary", "travel guide"],
        "academic": ["university", "college exam", "grade", "homework", "research", "thesis", "professor"],
        "other": []
    }
    
    for category, keywords in categories.items():
        for kw in keywords:
            if kw in text_lower:
                return category
    
    # Fallback: check for standalone irrelevant terms
    standalone_irrelevant = [
        ("sports", ["cricket", "football", "soccer", "basketball", "tennis", "golf", "match", "game"]),
        ("politics", ["election", "vote", "politician", "government", "law"]),
        ("finance", ["stock", "crypto", "investing", "trading"]),
        ("health", ["symptoms", "diagnosis", "treatment", "medical"]),
        ("travel", ["flight", "hotel booking", "vacation", "travel"]),
    ]
    
    for category, terms in standalone_irrelevant:
        # Only trigger if no product context
        if not has_product_review_context(text_lower):
            if any(f" {t} " in f" {text_lower} " or text_lower.startswith(t + " ") for t in terms):
                return category
    
    return "unknown_topic"


# ==============================================================================
# DOMAIN KEYWORDS (OPTIMIZED with frozensets)
# ==============================================================================
DOMAIN_KEYWORDS: dict[str, dict] = {
    "electronics": {
        "single": frozenset({
            "battery", "camera", "screen", "display", "charger", "charging", "usb",
            "processor", "cpu", "ram", "storage", "speaker", "audio", "bluetooth",
            "wifi", "5g", "lte", "sensor", "gps", "fingerprint", "gaming",
            "graphics", "gpu", "refresh", "oled", "lcd", "amoled",
            "pixel", "megapixels", "zoom", "lens", "aperture",
            "waterproof", "ip68", "headphone", "earphone", "touchscreen",
            "phone", "tablet", "laptop", "smartwatch", "earbuds",
        }),
        "phrases": frozenset({
            "battery life", "fast charging", "night mode", "face unlock",
            "wireless charging", "refresh rate", "portrait mode",
            "call quality", "signal strength", "heating issue",
        }),
        "weight": 1,
    },
    "clothing": {
        "single": frozenset({
            "fabric", "material", "cotton", "polyester", "size", "fit", "tight",
            "loose", "stretch", "breathable", "wash", "color",
            "fade", "shrink", "stitch", "seam", "pocket", "zipper", "button",
            "sleeve", "collar", "hem", "inseam", "waist", "hip",
            "jeans", "shirt", "dress", "jacket", "shoes", "boots",
        }),
        "phrases": frozenset({
            "true to size", "runs small", "runs large", "machine wash",
            "color faded", "shrunk after washing", "comfortable fit",
        }),
        "weight": 1,
    },
    "food": {
        "single": frozenset({
            "taste", "flavor", "fresh", "spicy", "sweet", "salty", "bitter",
            "sour", "crispy", "crunchy", "texture", "aroma", "portion",
            "serving", "calorie", "organic", "ingredients", "nutrition", "protein",
            "expired", "stale", "packaging", "ingredient",
        }),
        "phrases": frozenset({
            "expiry date", "best before", "shelf life",
            "taste great", "flavorful", "value for money",
        }),
        "weight": 1,
    },
    "furniture": {
        "single": frozenset({
            "assembly", "stable", "wobbly", "wood", "metal",
            "leather", "cushion", "ergonomic", "backrest", "armrest",
            "drawer", "shelf", "chair", "desk", "bedframe", "mattress",
            "legs", "surface", "scratch",
        }),
        "phrases": frozenset({
            "easy to assemble", "hard to assemble", "assembly instructions",
            "worth the price", "sturdy build", "comfortable seating",
        }),
        "weight": 1,
    },
    "beauty": {
        "single": frozenset({
            "moisturizer", "serum", "foundation", "concealer", "mascara",
            "eyeshadow", "lipstick", "blush", "primer", "breakout",
            "acne", "hypoallergenic", "skin", "cream", "lotion",
        }),
        "phrases": frozenset({
            "setting spray", "cruelty free", "long lasting", "skin tone",
            "easy to apply", "buildable coverage", "blends well",
        }),
        "weight": 1,
    },
    "home_appliances": {
        "single": frozenset({
            "noise", "filter", "capacity", "powerful",
            "efficient", "temperature", "timer", "automatic",
            "vacuum", "blender", "mixer", "coffeemaker", "microwave",
        }),
        "phrases": frozenset({
            "energy efficient", "noise level", "easy to clean",
            "worth the price", "powerful motor", "durable build",
        }),
        "weight": 1,
    },
}


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def detect_domain(text: str) -> str:
    words = tokenize(text)
    text_lower = text.lower()
    total_words = len(words)
    scores: dict[str, float] = {}

    for domain, config in DOMAIN_KEYWORDS.items():
        score = 0.0
        weight = config.get("weight", 1)
        
        single_hits = words & config["single"]
        score += len(single_hits) * weight * 1.5
        
        for phrase in config.get("phrases", set()):
            if phrase in text_lower:
                score += 3 * weight
        
        scores[domain] = score

    if not scores:
        return "generic"
    
    max_score = max(scores.values())
    normalized_threshold = max(1.5, total_words * 0.1)
    if max_score >= normalized_threshold:
        return max(scores, key=scores.get)
    return "generic"


# ==============================================================================
# NEGATION HANDLING
# ==============================================================================
SPECIAL_NEGATIONS = {
    "not bad": "decent", "not the worst": "acceptable", "not good": "poor",
    "not great": "mediocre", "not happy": "dissatisfied", "not disappointed": "satisfied",
    "not terrible": "acceptable", "not bad at all": "good", "not half bad": "decent",
    "nothing bad": "satisfactory", "no complaints about": "satisfactory",
    "can't complain": "acceptable", "couldn't be better": "excellent",
    "not a problem": "acceptable", "no problem with": "satisfied",
    "not disappointing": "adequate", "not the best": "mediocre",
    "not impressive": "mediocre", "not satisfied": "dissatisfied",
    "not worth": "overpriced", "not recommend": "avoid", "not recommended": "avoid",
    "wouldn't recommend": "avoid", "not buying again": "disappointed",
    "not happy with": "dissatisfied", "not impressed": "mediocre",
    "nothing special": "average", "not worth it": "overpriced",
    "not worth the money": "overpriced", "not a fan": "disappointed",
    "not comfortable": "uncomfortable", "not easy": "complicated",
    "not fast": "slow", "not quiet": "noisy", "not durable": "flimsy",
    "not sturdy": "flimsy", "not reliable": "unreliable", "not accurate": "inaccurate",
    "not sharp": "blurry", "not bright": "dim", "not smooth": "rough",
    "not clear": "unclear", "not responsive": "laggy", "not user friendly": "complicated",
}


def handle_special_negations(text: str) -> str:
    text_lower = text.lower()
    for phrase, replacement in SPECIAL_NEGATIONS.items():
        if phrase in text_lower:
            text = re.sub(r'\b' + re.escape(phrase) + r'\b', replacement, text, flags=re.IGNORECASE)
    return text


# ==============================================================================
# FEATURE DETECTION
# ==============================================================================
BASE_FEATURES = {
    "battery": frozenset({"battery", "backup", "drain", "mah", "power", "charging", "charge", "charger", "battery life", "drains"}),
    "camera": frozenset({"camera", "photo", "video", "lens", "focus", "zoom", "selfie", "megapixel", "aperture", "photos", "pictures"}),
    "performance": frozenset({"performance", "lag", "slow", "speed", "processor", "ram", "gaming", "gpu", "fast", "responsive"}),
    "design": frozenset({"design", "look", "style", "color", "aesthetic", "sleek", "premium", "appearance"}),
    "display": frozenset({"display", "screen", "brightness", "touch", "oled", "lcd", "amoled", "resolution", "visuals"}),
    "sound": frozenset({"sound", "audio", "speaker", "volume", "bass", "microphone", "mic", "call quality"}),
    "charging": frozenset({"charging", "charge", "charger", "wireless", "fast charging"}),
    "build": frozenset({"build", "quality", "durable", "material", "plastic", "metal", "glass", "waterproof", "sturdy", "flimsy"}),
    "price": frozenset({"price", "cost", "expensive", "value", "worth", "cheap", "affordable", "budget", "money"}),
    "software": frozenset({"software", "ui", "update", "app", "os", "android", "ios", "feature", "apps"}),
    "support": frozenset({"support", "service", "warranty", "help", "response", "customer"}),
    "comfort": frozenset({"comfort", "fit", "pain", "ear", "heavy", "light", "weight", "ergonomic", "comfortable"}),
    "connectivity": frozenset({"wifi", "bluetooth", "signal", "network", "5g", "lte", "gps", "connection"}),
    "delivery": frozenset({"delivery", "shipping", "arrived", "packaging", "damaged", "late", "on time", "fast", "slow"}),
}

DOMAIN_FEATURES = {
    "clothing": {
        "fabric": frozenset({"fabric", "material", "cotton", "polyester", "silk", "denim", "texture"}),
        "fit": frozenset({"fit", "size", "tight", "loose", "true to size", "runs small", "runs large"}),
        "comfort": frozenset({"comfortable", "soft", "breathable", "itchy", "rough", "comfort"}),
        "durability": frozenset({"durability", "fade", "shrink", "stretch", "tear", "pilling", "color faded"}),
    },
    "food": {
        "taste": frozenset({"taste", "flavor", "bland", "delicious", "savory", "sweet", "flavorful"}),
        "texture": frozenset({"texture", "crispy", "crunchy", "chewy", "tender", "dry", "moist", "fresh", "stale"}),
        "value": frozenset({"portion", "serving", "value", "fresh", "expired", "price"}),
    },
    "furniture": {
        "assembly": frozenset({"assembly", "instructions", "difficult", "assemble", "easy to assemble"}),
        "stability": frozenset({"stable", "wobbly", "sturdy", "solid", "flimsy", "shake"}),
        "comfort": frozenset({"comfortable", "cushion", "support", "firm", "soft", "seat", "back"}),
    },
    "beauty": {
        "application": frozenset({"application", "blend", "coverage", "pigmented", "patchy", "apply", "easy to apply"}),
        "wear": frozenset({"wear", "lasting", "smudge", "transfer", "fade", "settle", "long lasting"}),
        "skin": frozenset({"breakout", "irritation", "allergic", "sensitive", "oily", "dry", "skin"}),
    },
    "generic": {
        "quality": frozenset({"quality", "durable", "cheap", "premium", "sturdy", "flimsy", "solid"}),
        "usability": frozenset({"easy", "difficult", "convenient", "complicated", "intuitive", "user friendly"}),
        "value": frozenset({"value", "worth", "price", "expensive", "affordable", "budget", "money"}),
        "packaging": frozenset({"packaging", "arrived", "damaged", "sealed", "wrapped", "shipping"}),
        "delivery": frozenset({"delivery", "shipping", "arrived", "late", "on time", "fast", "slow"}),
    },
}


def get_features_for_domain(domain: str) -> dict:
    features = copy.deepcopy(BASE_FEATURES)
    if domain in DOMAIN_FEATURES:
        features.update(DOMAIN_FEATURES[domain])
    elif domain == "generic":
        features.update(DOMAIN_FEATURES["generic"])
    return features


# ==============================================================================
# AHO-CORASICK AUTOMATONS
# ==============================================================================
_ALIAS_AUTOMATONS: dict[str, 'ahocorasick.Automaton'] = {}
_PHRASE_AUTOMATONS: dict[str, 'ahocorasick.Automaton'] = {}
_ALIAS_TO_FEATURE: dict[str, dict[str, str]] = {}
_AHOCORASICK_BUILT = False


def _build_ahocorasick_automaton(domain: str) -> Tuple['ahocorasick.Automaton', 'ahocorasick.Automaton', dict]:
    features = get_features_for_domain(domain)
    
    word_automaton = ahocorasick.Automaton()
    word_to_feature = {}
    
    phrase_automaton = ahocorasick.Automaton()
    phrase_to_feature = {}
    
    for feature, aliases in features.items():
        for alias in aliases:
            if " " in alias:
                phrase_automaton.add_word(alias, (alias, feature))
                phrase_to_feature[alias] = feature
            else:
                word_automaton.add_word(alias, (alias, feature))
                word_to_feature[alias] = feature
    
    word_automaton.make_automaton()
    phrase_automaton.make_automaton()
    
    return word_automaton, phrase_automaton, word_to_feature


def _build_automaton_maps():
    global _ALIAS_AUTOMATONS, _PHRASE_AUTOMATONS, _ALIAS_TO_FEATURE, _AHOCORASICK_BUILT
    if _AHOCORASICK_BUILT:
        return
    
    for domain in ["electronics", "clothing", "food", "furniture", "beauty", "home_appliances", "generic"]:
        word_auto, phrase_auto, word_map = _build_ahocorasick_automaton(domain)
        _ALIAS_AUTOMATONS[domain] = word_auto
        _PHRASE_AUTOMATONS[domain] = phrase_auto
        _ALIAS_TO_FEATURE[domain] = word_map
    
    _AHOCORASICK_BUILT = True
    logger.info("Aho-Corasick automatons built for all domains")


# ==============================================================================
# SENTIMENT WORDS
# ==============================================================================
STRONG_NEGATIVE = frozenset({
    "bad", "poor", "worst", "waste", "overheat", "lag", "drain", "heats",
    "fails", "failure", "crash", "buggy", "terrible", "horrible", "awful",
    "broken", "disappointing", "frustrating", "useless", "defective",
    "cheaply", "flimsy", "pathetic", "regret", "nightmare", "disaster",
    "avoid", "refuse", "struggles", "slow", "drains", "weak", "unreliable",
    "damaged", "overheating", "overheats", "glitchy", "laggy", "unresponsive",
    "cheap", "brittle", "shoddy", "subpar", "underwhelming",
    "inconsistent", "unstable", "wobbly", "scratches", "scratchy",
    "stains", "faded", "shrunk", "ripped", "leaked", "leaking",
    "stopped working", "died", "dead", "mediocre",
})

STRONG_POSITIVE = frozenset({
    "excellent", "great", "smooth", "easy", "premium", "bright",
    "sharp", "clean", "love", "best", "amazing", "perfect", "outstanding",
    "fantastic", "wonderful", "brilliant", "superb", "impressed",
    "recommend", "exceeded", "delighted", "flawless", "exceptional",
    "remarkable", "solid", "reliable",
})

SOFT_NEGATIVE = frozenset({
    "slow", "weak", "issue", "problem", "expensive", "overpriced", "noisy", "hot",
    "average", "mediocre", "meh", "uncomfortable", "inconvenient", "complicated",
})

SOFT_POSITIVE = frozenset({
    "good", "nice", "fast", "clear", "lightweight", "sleek", "fine", "okay", "decent",
    "satisfactory", "acceptable", "adequate", "pleasant",
})

ALL_POSITIVE = STRONG_POSITIVE | SOFT_POSITIVE
ALL_NEGATIVE = STRONG_NEGATIVE | SOFT_NEGATIVE

_STRONG_POS_PATTERN = re.compile(r'\b(' + '|'.join(sorted(STRONG_POSITIVE, key=len, reverse=True)) + r')\b')
_SOFT_POS_PATTERN = re.compile(r'\b(' + '|'.join(sorted(SOFT_POSITIVE, key=len, reverse=True)) + r')\b')
_STRONG_NEG_PATTERN = re.compile(r'\b(' + '|'.join(sorted(STRONG_NEGATIVE, key=len, reverse=True)) + r')\b')
_SOFT_NEG_PATTERN = re.compile(r'\b(' + '|'.join(sorted(SOFT_NEGATIVE, key=len, reverse=True)) + r')\b')

_SENTIMENT_AUTOMATON = None
_SENTIMENT_WORD_MAP = {}


def _build_sentiment_automaton():
    global _SENTIMENT_AUTOMATON, _SENTIMENT_WORD_MAP
    
    auto = ahocorasick.Automaton()
    
    for word in ALL_POSITIVE:
        auto.add_word(word, ("positive", word))
    for word in ALL_NEGATIVE:
        auto.add_word(word, ("negative", word))
    
    auto.make_automaton()
    _SENTIMENT_AUTOMATON = auto
    
    _SENTIMENT_WORD_MAP = {word: "positive" for word in ALL_POSITIVE}
    _SENTIMENT_WORD_MAP.update({word: "negative" for word in ALL_NEGATIVE})


@lru_cache(maxsize=5000)
def get_sentiment_polarity_cached(text: str) -> Tuple[float, float]:
    text = handle_special_negations(text)
    text_lower = text.lower()
    
    vader_score = 0.0
    vader_confidence = 0.0
    
    if USE_VADER:
        vader_result = vader_analyzer.polarity_scores(text)
        vader_score = vader_result["compound"]
        vader_confidence = abs(vader_score)
        
        pos = vader_result["pos"]
        neg = vader_result["neg"]
        neu = vader_result["neu"]
        
        if max(pos, neg, neu) < 0.5:
            vader_confidence *= 0.7
        elif pos > 0.3 and neg > 0.3:
            vader_confidence *= 0.6
    
    if _SENTIMENT_AUTOMATON:
        keyword_counts = {"positive": 0, "negative": 0}
        for end_idx, (sentiment, word) in _SENTIMENT_AUTOMATON.iter(text_lower):
            if sentiment == "positive":
                keyword_counts["positive"] += 1
            else:
                keyword_counts["negative"] += 1
        kw_pos = keyword_counts["positive"]
        kw_neg = keyword_counts["negative"]
    else:
        kw_pos = len(_STRONG_POS_PATTERN.findall(text_lower)) + len(_SOFT_POS_PATTERN.findall(text_lower))
        kw_neg = len(_STRONG_NEG_PATTERN.findall(text_lower)) + len(_SOFT_NEG_PATTERN.findall(text_lower))
    
    kw_score = 0.0
    kw_confidence = min(1.0, (kw_pos + kw_neg) / 5)
    
    for _ in range(kw_pos):
        kw_score += 0.2
    for _ in range(kw_neg):
        kw_score -= 0.2
    
    words = text_lower.split()
    for i, word in enumerate(words):
        if word in _SENTIMENT_WORD_MAP:
            if i > 0 and words[i-1] in NEGATIONS:
                if _SENTIMENT_WORD_MAP[word] == "positive":
                    kw_score -= 0.3
                else:
                    kw_score += 0.3
    
    if vader_confidence > 0.7:
        weight_vader = 0.75
        weight_kw = 0.25
    elif vader_confidence > 0.4:
        weight_vader = 0.6
        weight_kw = 0.4
    else:
        weight_vader = 0.4
        weight_kw = 0.6
    
    if kw_confidence > 0.8:
        weight_kw = min(0.7, weight_kw + 0.1)
        weight_vader = 1.0 - weight_kw
    
    blended = weight_vader * vader_score + weight_kw * kw_score
    overall_confidence = max(vader_confidence, kw_confidence)
    
    return max(-1.0, min(1.0, blended)), overall_confidence


def get_sentiment_polarity(text: str) -> float:
    polarity, _ = get_sentiment_polarity_cached(text.lower().strip())
    return polarity


def get_sentiment_confidence(text: str) -> float:
    _, confidence = get_sentiment_polarity_cached(text.lower().strip())
    return confidence


def classify_sentence(sentence: str) -> str:
    polarity = get_sentiment_polarity(sentence)
    if polarity > SENTIMENT_POLARITY_THRESHOLD:
        return "positive"
    if polarity < -SENTIMENT_POLARITY_THRESHOLD:
        return "negative"
    return "neutral"


# ==============================================================================
# FEATURE EXTRACTION
# ==============================================================================
def extract_features_with_context(sentence: str, domain: str = "generic") -> List[str]:
    sentence_lower = sentence.lower()
    detected_features = set()
    
    if domain in _PHRASE_AUTOMATONS:
        for end_idx, (phrase, feature) in _PHRASE_AUTOMATONS[domain].iter(sentence_lower):
            detected_features.add(feature)
    
    if domain in _ALIAS_AUTOMATONS:
        for end_idx, (word, feature) in _ALIAS_AUTOMATONS[domain].iter(sentence_lower):
            detected_features.add(feature)
    
    return list(detected_features) if detected_features else []


@lru_cache(maxsize=1000)
def extract_feature_cached(sentence: str, domain: str = "generic") -> Optional[str]:
    features = extract_features_with_context(sentence, domain)
    return features[0] if features else None


# ==============================================================================
# CLAUSE SPLITTING
# ==============================================================================
_CONNECTOR_PATTERN = re.compile(r'\s+((?:but|however|although|though|while|whereas|yet|except|otherwise|nonetheless|nevertheless|alternatively|instead|also|plus|and then))\s+', re.I)
_CONJUNCTION_PATTERN = re.compile(r',\s*(?=(but|however|although|though|and also|plus|and then))\s+', re.I)
_PUNCTUATION_SPLIT = re.compile(r'(?<=[.!?])\s+')


def split_into_clauses(text: str) -> list[dict]:
    segments = []
    
    sentences = _PUNCTUATION_SPLIT.split(text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        parts = _CONJUNCTION_PATTERN.split(sentence)
        
        current_connector = None
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            connector_match = _CONNECTOR_PATTERN.match(part)
            if connector_match:
                current_connector = connector_match.group(1).lower()
                part = _CONNECTOR_PATTERN.sub('', part, count=1).strip()
            
            if not part:
                continue
            
            for prefix in ("but ", "however ", "although ", "though ", "while ", "yet ", "except "):
                if part.lower().startswith(prefix):
                    part = part[len(prefix):].strip()
                    break
            
            if part and part[0].islower():
                part = part[0].upper() + part[1:]
            
            part = part.strip(" ,.;")
            
            if part and len(part) >= 5:
                segments.append({
                    "text": part,
                    "connector": current_connector
                })
            
            current_connector = None
    
    return segments


def shorten(text: str, max_length: int = 80) -> str:
    if not text:
        return text
    text = text.strip()
    if "," in text:
        text = text.split(",")[0]
    elif "." in text:
        text = text.split(".")[0]
    if len(text) > max_length:
        text = text[: max_length - 3].rstrip() + "..."
    return text.strip()


# ==============================================================================
# PYDANTIC MODELS
# ==============================================================================
class ReviewRequest(BaseModel):
    reviews: list[str] = Field(..., min_length=1)

    @field_validator("reviews")
    @classmethod
    def clean_reviews(cls, reviews: list[str]) -> list[str]:
        cleaned = [r.strip() for r in reviews if r and len(r.strip()) >= 5]
        if not cleaned:
            raise ValueError("Please provide at least one review with content.")
        if len(cleaned) > MAX_REVIEWS:
            raise ValueError(f"Please provide no more than {MAX_REVIEWS} reviews.")
        return cleaned


class RawAnalyzeRequest(BaseModel):
    raw_text: str = Field(..., min_length=10)
    user_focus: Optional[str] = Field(None)

    @field_validator("raw_text")
    @classmethod
    def validate_size(cls, v: str) -> str:
        if len(v) > MAX_INPUT_SIZE:
            raise ValueError(f"Input too large. Max {MAX_INPUT_SIZE} chars.")
        return v


class SentimentBreakdown(BaseModel):
    positive: float
    neutral: float
    negative: float
    total: int


class ExplainablePoint(BaseModel):
    text: str
    features: List[str]
    sentiment: str
    polarity_score: float
    confidence: float
    impact: str


class AnalysisPoint(BaseModel):
    text: str
    feature: str
    features: List[str] = Field(default_factory=list)
    impact: str


class FeatureScore(BaseModel):
    feature: str
    display_name: str
    positive_count: int
    negative_count: int
    total_mentions: int
    score: float


class AnalyzeResponse(BaseModel):
    summary: str
    pros: list[AnalysisPoint]
    cons: list[AnalysisPoint]
    neutral_points: list[str] = Field(default_factory=list)
    sentiment: SentimentBreakdown
    score: float
    confidence: float
    cached: bool = False
    warnings: list[WarningDetail] = Field(default_factory=list)
    explained_pros: list[ExplainablePoint] | None = None
    explained_cons: list[ExplainablePoint] | None = None
    feature_scores: list[FeatureScore] | None = None
    domain: str | None = None
    out_of_scope: bool = False  # NEW: Flag for out-of-scope input


# ==============================================================================
# FASTAPI APP
# ==============================================================================
app = FastAPI(
    title="AI Product Review Aggregator API",
    description="Production-ready system with FULL AI-driven analysis and out-of-scope detection",
    version="25.0-ai-validated",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    if not VALID_API_KEYS:
        logger.warning("No API_KEYS configured - running in open mode")
        return "dev"
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required. Pass 'X-API-Key' header.")
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return x_api_key


# ==============================================================================
# PROMETHEUS METRICS
# ==============================================================================
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    REQUEST_COUNT = Counter("review_api_requests_total", "Total requests", ["endpoint", "status"])
    REQUEST_LATENCY = Histogram("review_api_request_duration_seconds", "Request latency", ["endpoint"])
    CACHE_HITS = Counter("review_api_cache_hits_total", "Cache hits")
    ACTIVE_REQUESTS = Gauge("review_api_active_requests", "Active requests")
    GEMINI_REQUESTS_ACTIVE = Gauge("review_api_gemini_active", "Active Gemini requests")
    OUT_OF_SCOPE_COUNT = Counter("review_api_out_of_scope_total", "Out of scope requests")
    USE_PROMETHEUS = True

    @app.get("/metrics")
    async def metrics():
        return JSONResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
except ImportError:
    USE_PROMETHEUS = False
    class NoOpCounter:
        def __init__(self, *_, **__): pass
        def labels(self, **_): return self
        def inc(self, *_, **__): pass
    class NoOpHistogram:
        def __init__(self, *_, **__): pass
        def labels(self, **_):
            class Ctx:
                def __enter__(self): return self
                def __exit__(self, *_, **__): pass
                def time(self): return self
            return Ctx()
    class NoOpGauge:
        def __init__(self, *_, **__): pass
        def labels(self, **_): return self
        def set(self, *_): pass
    REQUEST_COUNT = NoOpCounter()
    REQUEST_LATENCY = NoOpHistogram()
    CACHE_HITS = NoOpCounter()
    ACTIVE_REQUESTS = NoOpGauge()
    GEMINI_REQUESTS_ACTIVE = NoOpGauge()
    OUT_OF_SCOPE_COUNT = NoOpCounter()


# ==============================================================================
# REQUEST TRACKER
# ==============================================================================
class RequestTracker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._active_http = 0
                    cls._instance._active_gemini = 0
        return cls._instance

    @property
    def active_http(self) -> int:
        return self._active_http

    @active_http.setter
    def active_http(self, value: int):
        self._active_http = max(0, value)
        ACTIVE_REQUESTS.set(self._active_http)

    @property
    def active_gemini(self) -> int:
        return self._active_gemini

    @active_gemini.setter
    def active_gemini(self, value: int):
        self._active_gemini = max(0, value)
        GEMINI_REQUESTS_ACTIVE.set(self._active_gemini)


request_tracker = RequestTracker()


# ==============================================================================
# LRU CACHE
# ==============================================================================
class LRUCache:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.size = 0

    def _serialize(self, data: dict) -> str:
        if USE_ORJSON:
            return orjson.dumps(data).decode('utf-8')
        return json.dumps(data)

    def _deserialize(self, data_str: str) -> dict:
        if USE_ORJSON:
            return orjson.loads(data_str)
        return json.loads(data_str)

    def get(self, key: str) -> Optional[dict]:
        if key not in self.cache:
            self.miss_count += 1
            return None
        data_str, expiry = self.cache[key]
        if time.time() > expiry:
            del self.cache[key]
            self.size -= 1
            self.miss_count += 1
            return None
        self.cache.move_to_end(key)
        self.hit_count += 1
        CACHE_HITS.inc()
        return self._deserialize(data_str)

    def set(self, key: str, data: dict, ttl: int = 3600):
        while self.size >= self.max_size:
            self.cache.popitem(last=False)
            self.size -= 1
        data_str = self._serialize(data)
        self.cache[key] = (data_str, time.time() + ttl)
        self.cache.move_to_end(key)
        self.size += 1

    def get_stats(self) -> dict:
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        return {"hits": self.hit_count, "misses": self.miss_count, "hit_rate": f"{hit_rate:.1f}%"}


class CacheManager:
    def __init__(self):
        self.lru_cache = LRUCache(max_size=MAX_CACHE_SIZE)

    def generate_cache_key(self, reviews: list[str], detailed: bool = False, domain: str = "generic") -> str:
        normalized_reviews = []
        for r in reviews:
            cleaned = r.strip().lower()
            cleaned = re.sub(r'\s+', ' ', cleaned)
            normalized_reviews.append(cleaned)
        
        content_hash = hashlib.sha256("|".join(normalized_reviews).encode()).hexdigest()[:24]
        
        payload = {"reviews": normalized_reviews, "detailed": detailed, "domain": domain, "hash": content_hash}
        cache_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return f"review_cache:{cache_hash}"

    def get(self, key: str) -> Optional[dict]:
        if not ENABLE_CACHE:
            return None
        return self.lru_cache.get(key)

    def set(self, key: str, data: dict, ttl: int = None):
        if not ENABLE_CACHE:
            return
        ttl = ttl or CACHE_TTL_SECONDS
        self.lru_cache.set(key, data, ttl)


cache_manager = CacheManager()


# ==============================================================================
# RATE LIMITER
# ==============================================================================
class ScalableRateLimiter:
    def __init__(self):
        self._requests_per_minute: dict[str, deque] = {}
        self._requests_per_hour: dict[str, deque] = {}
        self._lock = threading.Lock()
        self._cleanup_interval = 300
        self._last_cleanup = time.time()

    def _cleanup_old_entries(self):
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        with self._lock:
            cutoff_minute = now - 60
            cutoff_hour = now - 3600
            
            for key in list(self._requests_per_minute.keys()):
                while self._requests_per_minute[key] and self._requests_per_minute[key][0] < cutoff_minute:
                    self._requests_per_minute[key].popleft()
                if not self._requests_per_minute[key]:
                    del self._requests_per_minute[key]
            
            for key in list(self._requests_per_hour.keys()):
                while self._requests_per_hour[key] and self._requests_per_hour[key][0] < cutoff_hour:
                    self._requests_per_hour[key].popleft()
                if not self._requests_per_hour[key]:
                    del self._requests_per_hour[key]
            
            self._last_cleanup = now

    def record_request(self, identifier: str):
        now = time.time()
        self._cleanup_old_entries()
        
        with self._lock:
            if identifier not in self._requests_per_minute:
                self._requests_per_minute[identifier] = deque(maxlen=RATE_LIMIT_PER_MINUTE + 10)
            if identifier not in self._requests_per_hour:
                self._requests_per_hour[identifier] = deque(maxlen=RATE_LIMIT_PER_HOUR + 10)
            
            self._requests_per_minute[identifier].append(now)
            self._requests_per_hour[identifier].append(now)

    def get_request_count(self, identifier: str, window_seconds: int = 60) -> int:
        now = time.time()
        with self._lock:
            if window_seconds <= 60:
                if identifier not in self._requests_per_minute:
                    return 0
                cutoff = now - 60
                while self._requests_per_minute[identifier] and self._requests_per_minute[identifier][0] < cutoff:
                    self._requests_per_minute[identifier].popleft()
                return len(self._requests_per_minute[identifier])
            else:
                if identifier not in self._requests_per_hour:
                    return 0
                cutoff = now - 3600
                while self._requests_per_hour[identifier] and self._requests_per_hour[identifier][0] < cutoff:
                    self._requests_per_hour[identifier].popleft()
                return len(self._requests_per_hour[identifier])

    def is_rate_limited(self, identifier: str, per_minute: int = None, per_hour: int = None) -> tuple[bool, str]:
        per_minute = per_minute or RATE_LIMIT_PER_MINUTE
        per_hour = per_hour or RATE_LIMIT_PER_HOUR
        minute_count = self.get_request_count(identifier, 60)
        if minute_count >= per_minute:
            return True, f"Per-minute limit exceeded ({minute_count}/{per_minute})"
        hour_count = self.get_request_count(identifier, 3600)
        if hour_count >= per_hour:
            return True, f"Per-hour limit exceeded ({hour_count}/{per_hour})"
        return False, ""


scalable_limiter = ScalableRateLimiter()


# ==============================================================================
# GEMINI CLIENT MANAGER
# ==============================================================================
class GeminiKeyConfig:
    def __init__(self, key: str, name: str = ""):
        self._key = key.strip()
        self._hash = hashlib.sha256(self._key.encode()).hexdigest()
        self._name = name or f"key_{self._hash[:6]}"
        self._is_available = True
        self._failure_count = 0
        self._last_failure: Optional[float] = None
        self._success_count = 0
        self._total_requests = 0
        self._failure_threshold = 5
        self._recovery_timeout = 300

    @property
    def key(self) -> str:
        return self._key

    @property
    def id(self) -> str:
        return self._hash[:6]

    @property
    def is_healthy(self) -> bool:
        if not self._is_available:
            if self._last_failure and (time.time() - self._last_failure) > self._recovery_timeout:
                self._is_available = True
                self._failure_count = 0
                return True
            return False
        return True

    def record_success(self):
        self._success_count += 1
        self._total_requests += 1
        if self._failure_count > 0:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self, is_rate_limit: bool = False):
        self._failure_count += 1
        self._last_failure = time.time()
        self._total_requests += 1
        if self._failure_count >= self._failure_threshold or is_rate_limit:
            self._is_available = False


class GeminiClientManager:
    _instance: Optional["GeminiClientManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._keys: dict[str, GeminiKeyConfig] = {}
        self._rotation_lock = threading.Lock()
        self._index = 0
        self._round_robin: list[str] = []
        self._max_retries = 2
        self._consecutive_failures = 0
        self._circuit_open_time: Optional[float] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._load_keys()
        healthy = self.get_healthy_key_count()
        logger.info(f"GeminiClientManager: {len(self._keys)} keys, {healthy} healthy")

    def has_keys(self) -> bool:
        return bool(self._keys)

    def get_healthy_key_count(self) -> int:
        return sum(1 for cfg in self._keys.values() if cfg.is_healthy)

    def is_circuit_open(self) -> bool:
        if self._circuit_open_time is None:
            return False
        if time.time() - self._circuit_open_time > 60:
            self._circuit_open_time = None
            self._consecutive_failures = 0
            return False
        return True

    def _load_keys(self):
        self._keys.clear()
        raw_keys = os.getenv("GEMINI_API_KEYS", "")
        single_key = os.getenv("GEMINI_API_KEY", "").strip()
        existing_hashes: set[str] = set()

        if raw_keys:
            for entry in raw_keys.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                if ":" in entry:
                    name, key = entry.split(":", 1)
                    self._add_key(key.strip(), name.strip(), existing_hashes)
                else:
                    self._add_key(entry, "", existing_hashes)
        elif single_key and single_key not in PLACEHOLDER_API_KEYS:
            self._add_key(single_key, "default", existing_hashes)
        self._rebuild_rotation()

    def _add_key(self, key: str, name: str, existing_hashes: set[str]):
        if not key:
            return
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        if key_hash in existing_hashes:
            return
        existing_hashes.add(key_hash)
        self._keys[key_hash] = GeminiKeyConfig(key, name)

    def _rebuild_rotation(self):
        with self._rotation_lock:
            healthy = [h for h, cfg in self._keys.items() if cfg.is_healthy]
            if healthy:
                random.shuffle(healthy)
                self._round_robin = healthy
            else:
                self._round_robin = list(self._keys.keys())
            self._index = 0

    def _get_next_key(self) -> Optional[tuple[GeminiKeyConfig, str]]:
        with self._rotation_lock:
            if not self._round_robin or self.is_circuit_open():
                return None
            for _ in range(len(self._round_robin)):
                key_hash = self._round_robin[self._index]
                self._index = (self._index + 1) % len(self._round_robin)
                cfg = self._keys.get(key_hash)
                if cfg and cfg.is_healthy:
                    return cfg, key_hash
            self._rebuild_rotation()
            if not self._round_robin:
                return None
            key_hash = self._round_robin[self._index % len(self._round_robin)]
            cfg = self._keys.get(key_hash)
            return (cfg, key_hash) if cfg else None

    async def _init_http_client(self):
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=GEMINI_TIMEOUT)
        return self._http_client

    async def analyze_reviews(self, reviews: list[str], model_name: str = DEFAULT_GEMINI_MODEL, domain: str = "generic") -> Optional[dict]:
        prompt = build_analysis_prompt(reviews, domain)

        nxt = self._get_next_key()
        if not nxt:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3 and self._circuit_open_time is None:
                self._circuit_open_time = time.time()
            return None

        key_cfg, _ = nxt
        
        try:
            client = await self._init_http_client()
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key_cfg.key}"
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2}
            }
            
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                key_cfg.record_success()
                self._consecutive_failures = 0
                data = response.json()
                if "candidates" in data and data["candidates"]:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    return parse_ai_response(text)
            elif response.status_code == 429:
                key_cfg.record_failure(is_rate_limit=True)
                self._consecutive_failures += 1
            else:
                key_cfg.record_failure()
                self._consecutive_failures += 1
            
            return None
            
        except httpx.TimeoutException:
            key_cfg.record_failure()
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3 and self._circuit_open_time is None:
                self._circuit_open_time = time.time()
            return None
        except Exception as e:
            logger.warning(f"Gemini API error: {e}")
            key_cfg.record_failure()
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3 and self._circuit_open_time is None:
                self._circuit_open_time = time.time()
            return None

    async def close(self):
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def get_health(self) -> dict:
        return {
            "total_keys": len(self._keys),
            "healthy_keys": self.get_healthy_key_count(),
            "circuit_open": self.is_circuit_open(),
        }


gemini_manager = GeminiClientManager()


# ==============================================================================
# SENTIMENT & ANALYSIS FUNCTIONS
# ==============================================================================
def parse_raw_input(raw_text: str) -> list[str]:
    if not raw_text:
        return []
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    reviews = []

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for prefix_pattern in [
            r'^[\u2022\u2023\u25E6\u2043\u2219\*\-\+]+\s*',
            r'^\d+[\).\]\:]+\s*',
            r'^[""\'"]+',
            r'^\*\*(.+?)\*\*:',
            r'^[A-Za-z\s]+:\s*',
        ]:
            line = re.sub(prefix_pattern, "", line, flags=re.IGNORECASE)
        line = line.strip('"\' -')
        if len(line) < 5:
            continue
        if len(line) > 150:
            for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line):
                part = part.strip()
                if len(part) >= 10:
                    reviews.append(part)
        else:
            reviews.append(line)

    if len(reviews) < 3 and len(raw_text) > 200:
        for part in re.split(r'\.(?=\s|$)', raw_text):
            part = part.strip()
            if len(part) >= 10:
                is_new = True
                for existing in reviews[:20]:
                    if SequenceMatcher(None, part.lower(), existing.lower()).ratio() > 0.8:
                        is_new = False
                        break
                if is_new:
                    reviews.append(part)

    return [r.strip() for r in reviews if len(r.strip()) >= 5][:MAX_REVIEWS]


def is_valid_fragment(text: str) -> bool:
    if len(text) < 5 or len(text.split()) < 1:
        return False
    if re.search(r"(.)\1{5,}", text):
        return False
    return True


def normalize_point(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" -*\t\r\n")
    cleaned = re.sub(r"^\d+[\).\s-]+", "", cleaned)
    
    words = cleaned.lower().split()
    if len(words) <= 3 and any(w in FILLER_WORDS for w in words):
        return ""
    
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    if len(cleaned) > 120:
        cleaned = cleaned[:117].rstrip() + "..."
    return cleaned


def get_point_signature(text: str) -> str:
    norm = normalize_point(text).lower()
    norm = re.sub(r"[^a-z0-9 ]", "", norm)
    stop_words = frozenset({"a", "an", "and", "as", "at", "for", "from", "in", "is", "of", "the", "to", "very", "with"})
    tokens = [t for t in norm.split() if t not in stop_words]
    return " ".join(tokens)


def points_overlap(a: str, b: str) -> bool:
    norm_a = normalize_point(a).lower()
    norm_b = normalize_point(b).lower()
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True
    if norm_a in norm_b or norm_b in norm_a:
        return True
    if SequenceMatcher(None, norm_a, norm_b).ratio() >= 0.84:
        return True
    return False


def is_useless_point(text: str) -> bool:
    text_lower = text.lower()
    useless_phrases = ("no major", "no clear", "no complaints", "no cons", "no negatives",
                      "no issues", "none", "not mentioned", "n/a")
    return text_lower.startswith(useless_phrases) or any(
        phrase in text_lower for phrase in ("no complaints mentioned", "no major recurring")
    )


def prepare_and_split(reviews: list[str]) -> list[dict]:
    prepared = []
    for raw in reviews:
        clauses = split_into_clauses(raw)
        for clause_dict in clauses:
            if is_valid_fragment(clause_dict["text"]):
                prepared.append(clause_dict)
    return prepared[:MAX_REVIEWS_TO_ANALYZE]


def get_impact_level(polarity_score: float) -> str:
    abs_score = abs(polarity_score)
    if abs_score >= 0.5:
        return "high"
    if abs_score >= 0.2:
        return "medium"
    return "low"


def make_analysis_point(text: str, domain: str, connector: str = None) -> AnalysisPoint:
    features_list = extract_features_with_context(text, domain)
    primary_feature = features_list[0] if features_list else "general"
    polarity = get_sentiment_polarity(text)
    confidence = get_sentiment_confidence(text)
    impact = get_impact_level(polarity)
    return AnalysisPoint(
        text=shorten(text), 
        feature=primary_feature,
        features=features_list,
        impact=impact
    )


# ==============================================================================
# 🚨 AI PROMPT WITH OUT-OF-SCOPE DETECTION
# ==============================================================================
def build_analysis_prompt(reviews: list[str], domain: str = "generic") -> str:
    return f"""You are a strict product review analyzer.

⚠️ CRITICAL RULE: If the input is NOT about product reviews, you MUST return an error response.

OUT-OF-SCOPE EXAMPLES (return error for these):
- Sports queries: "Tell me about cricket match"
- Politics: "Who won the election?"
- Entertainment: "What movie should I watch?"
- Finance: "Is Bitcoin going up?"
- Health: "What are symptoms of flu?"
- Travel: "Best hotels in Paris"
- Recipes: "How to make pasta?"

PRODUCT REVIEW INDICATORS (only analyze if present):
- Reviews, ratings, stars, recommend
- Product features (battery, camera, quality, etc.)
- Pros/Cons, advantages/disadvantages
- Purchase experience, shipping, delivery
- Price/value assessment

Return JSON format:

FOR VALID PRODUCT REVIEWS:
{{
  "summary": "...",
  "pros": [],
  "cons": [],
  "neutral_points": [],
  "sentiment": {{"positive": 0, "negative": 0, "neutral": 0}}
}}

FOR OUT-OF-SCOPE (non-product-review input):
{{
  "error": "OUT_OF_SCOPE",
  "reason": "Specific reason why this isn't a product review",
  "detected_category": "sports/politics/entertainment/finance/etc."
}}

Reviews:
{chr(10).join(reviews)}"""


# ==============================================================================
# AI HELPERS
# ==============================================================================
def resolve_model_name(raw: str) -> str:
    name = raw.strip() or DEFAULT_GEMINI_MODEL
    return LEGACY_MODEL_ALIASES.get(name, name)


def parse_ai_response(raw_text: str) -> Optional[dict]:
    """Parse AI response, including error responses for out-of-scope"""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()

    # Try to extract JSON from markdown code blocks first
    json_match = re.search(r'\{[\s\S]*\}', cleaned)
    if json_match:
        try:
            payload = json.loads(json_match.group())
            if isinstance(payload, dict):
                # Check if AI returned an error (out-of-scope)
                if "error" in payload and payload.get("error") == "OUT_OF_SCOPE":
                    return {
                        "out_of_scope": True,
                        "reason": payload.get("reason", "Not a product review"),
                        "detected_category": payload.get("detected_category", "unknown"),
                    }
                return _process_ai_payload(payload)
        except json.JSONDecodeError:
            pass

    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            # Check if AI returned an error (out-of-scope)
            if "error" in payload and payload.get("error") == "OUT_OF_SCOPE":
                return {
                    "out_of_scope": True,
                    "reason": payload.get("reason", "Not a product review"),
                    "detected_category": payload.get("detected_category", "unknown"),
                }
            return _process_ai_payload(payload)
    except (json.JSONDecodeError, Exception):
        pass

    # Fallback to regex extraction
    return _extract_ai_response_regex(cleaned)


def _process_ai_payload(payload: dict) -> Optional[dict]:
    """Process AI response payload and extract sentiment"""
    pros = payload.get("pros", [])
    cons = payload.get("cons", [])
    
    if not isinstance(pros, list) or not isinstance(cons, list):
        return None
    if not pros and not cons:
        return None
    
    ai_sentiment = payload.get("sentiment", {})
    if isinstance(ai_sentiment, dict):
        pos = ai_sentiment.get("positive", 0)
        neg = ai_sentiment.get("negative", 0)
        neu = ai_sentiment.get("neutral", 0)
    else:
        total = len(pros) + len(cons)
        pos = round((len(pros) / total) * 100, 2) if total else 0
        neg = round((len(cons) / total) * 100, 2) if total else 0
        neu = 0
    
    return {
        "summary": str(payload.get("summary", "")).strip(),
        "pros": [shorten(p) if isinstance(p, str) else shorten(p.get("text", str(p))) for p in pros if (isinstance(p, str) and len(p.strip()) >= 5) or (isinstance(p, dict) and len(p.get("text", "").strip()) >= 5)],
        "cons": [shorten(c) if isinstance(c, str) else shorten(c.get("text", str(c))) for c in cons if (isinstance(c, str) and len(c.strip()) >= 5) or (isinstance(c, dict) and len(c.get("text", "").strip()) >= 5)],
        "neutral_points": [shorten(n) if isinstance(n, str) else shorten(n.get("text", str(n))) for n in payload.get("neutral_points", []) if (isinstance(n, str) and len(n.strip()) >= 5) or (isinstance(n, dict) and len(n.get("text", "").strip()) >= 5)],
        "ai_sentiment": {
            "positive": pos,
            "negative": neg,
            "neutral": neu,
        }
    }


def _extract_ai_response_regex(cleaned: str) -> Optional[dict]:
    """Fallback regex extraction for non-JSON AI responses"""
    sm = re.search(r"summary\s*:\s*(.+?)(?:\n\s*pros?\s*:|\Z)", cleaned, flags=re.I | re.S)
    pm = re.search(r"pros?\s*:\s*(.+?)(?:\n\s*cons?\s*:|\Z)", cleaned, flags=re.I | re.S)
    cm = re.search(r"cons?\s*:\s*(.+?)(?:\n\s*neutral|\Z)", cleaned, flags=re.I | re.S)

    def extract_list(text):
        if not text:
            return []
        return [shorten(l.strip()) for l in text.splitlines()
                if l.strip() and ":" not in l.lower() and len(l.strip()) >= 5]

    pros = extract_list(pm.group(1) if pm else "")
    cons = extract_list(cm.group(1) if cm else "")

    if not pros and not cons:
        return None
    
    return {"summary": sm.group(1).strip() if sm else "", "pros": pros, "cons": cons, "neutral_points": []}


def build_summary(pros: list[AnalysisPoint], cons: list[AnalysisPoint], sentiment: dict, domain: str = "generic") -> str:
    pos_pct = sentiment.get("positive", 0)
    neg_pct = sentiment.get("negative", 0)
    neu_pct = sentiment.get("neutral", 0)

    pro_features = list(dict.fromkeys(p.feature for p in pros if p.feature != "general"))[:2]
    con_features = list(dict.fromkeys(c.feature for c in cons if c.feature != "general"))[:1]

    def fmt(f: str) -> str:
        return f.replace("_", " ")

    if pos_pct >= 60:
        if pro_features:
            feat_str = " and ".join(fmt(f) for f in pro_features)
            suffix = f" Some report issues with the {fmt(con_features[0])}." if con_features else ""
            return f"Users generally love the {feat_str}.{suffix}"
        return "Most users are satisfied with this product."
    elif neg_pct >= 60:
        if con_features:
            feat_str = fmt(con_features[0])
            suffix = f" A few highlight the {fmt(pro_features[0])} as a positive." if pro_features else ""
            return f"Users report notable concerns with the {feat_str}.{suffix}"
        return "Most users are dissatisfied with this product."
    elif pos_pct > neg_pct + 15:
        if pro_features and con_features:
            return f"The {fmt(pro_features[0])} gets praise, though some users flag {fmt(con_features[0])} issues."
        return "Reviews lean positive with a few concerns."
    elif neg_pct > pos_pct + 15:
        if con_features and pro_features:
            return f"Concerns focus on {fmt(con_features[0])}, despite a decent {fmt(pro_features[0])}."
        return "Reviews lean negative with a few bright spots."
    elif neu_pct >= 50:
        return "Reviews are largely mixed — experiences vary significantly across users."
    else:
        if pro_features and con_features:
            return f"A balanced product — {fmt(pro_features[0])} stands out positively, but {fmt(con_features[0])} needs work."
        return f"Feedback is balanced ({pos_pct:.0f}% positive, {neg_pct:.0f}% negative)."


def apply_user_focus(points: list[AnalysisPoint], user_focus: Optional[str]) -> list[AnalysisPoint]:
    if not user_focus:
        return points
    focus = user_focus.lower().strip()
    boosted = [p for p in points if focus in p.feature.lower() or any(focus in f.lower() for f in p.features) or focus in p.text.lower()]
    rest = [p for p in points if p not in boosted]
    return boosted + rest


# ==============================================================================
# 🚨 PROCESS ANALYSIS WITH OUT-OF-SCOPE DETECTION
# ==============================================================================
async def process_analysis_task(
    reviews: list[str],
    detailed: bool,
    user_focus: Optional[str] = None,
) -> Tuple[dict, dict, list, bool, list, str]:
    """FULLY AI-DRIVEN with pre-validation and out-of-scope detection"""
    warnings = []
    start_time = time.time()

    raw_text = " ".join(reviews)
    domain = detect_domain(raw_text)

    # 🚨 STEP 1: PRE-VALIDATION - Check relevance BEFORE AI call
    is_relevant, relevance_score, relevance_type = is_product_review_related(raw_text)
    
    if not is_relevant:
        OUT_OF_SCOPE_COUNT.inc()
        category = detect_out_of_scope_category(raw_text)
        logger.warning(f"Out-of-scope input detected: score={relevance_score:.2f}, type={relevance_type}, category={category}")
        
        return (
            {
                "summary": "",
                "pros": [],
                "cons": [],
                "neutral_points": [],
                "out_of_scope": True,
            },
            {"positive": 0, "negative": 0, "neutral": 0, "total": 0, "avg_confidence": 0.0},
            [],
            False,
            [WarningDetail(type="OUT_OF_SCOPE", message=f"Input does not appear to be a product review. Detected as: {category}. Relevance score: {relevance_score:.2f}")],
            domain,
        )

    # STEP 2: Cache check (only for relevant input)
    cache_key = cache_manager.generate_cache_key(reviews, detailed, domain)
    cached = cache_manager.get(cache_key)

    if cached:
        return (
            cached["analysis"],
            cached["sentiment"],
            cached.get("feature_scores", []),
            True,
            [],
            domain,
        )

    # STEP 3: AI call (only for relevant input)
    model_name = resolve_model_name(os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))

    request_tracker.active_gemini += 1
    try:
        ai_result = await gemini_manager.analyze_reviews(reviews, model_name, domain)
    finally:
        request_tracker.active_gemini -= 1

    # 🚨 STEP 4: Check if AI detected out-of-scope
    if ai_result and ai_result.get("out_of_scope"):
        OUT_OF_SCOPE_COUNT.inc()
        category = ai_result.get("detected_category", "unknown")
        reason = ai_result.get("reason", "Not a product review")
        
        return (
            {
                "summary": "",
                "pros": [],
                "cons": [],
                "neutral_points": [],
                "out_of_scope": True,
            },
            {"positive": 0, "negative": 0, "neutral": 0, "total": 0, "avg_confidence": 0.0},
            [],
            False,
            [WarningDetail(type="OUT_OF_SCOPE", message=f"AI detected non-product-review input: {reason}. Category: {category}")],
            domain,
        )

    if not ai_result:
        raise Exception("AI analysis failed completely")

    # STEP 5: Convert AI → schema
    pros = [make_analysis_point(p, domain) for p in ai_result.get("pros", [])]
    cons = [make_analysis_point(c, domain) for c in ai_result.get("cons", [])]
    neutral_points = ai_result.get("neutral_points", [])

    ai_sentiment = ai_result.get("ai_sentiment", {})
    total = len(pros) + len(cons) + len(neutral_points)

    if ai_sentiment:
        pos = ai_sentiment.get("positive", 0)
        neg = ai_sentiment.get("negative", 0)
        neu = ai_sentiment.get("neutral", 0)
    else:
        pos = len(pros)
        neg = len(cons)
        neu = len(neutral_points)

    sentiment = {
        "positive": round((pos / total) * 100, 2) if total else 0,
        "negative": round((neg / total) * 100, 2) if total else 0,
        "neutral": round((neu / total) * 100, 2) if total else 0,
        "total": total,
        "avg_confidence": 0.85
    }

    score, confidence = calculate_score_and_confidence(sentiment)

    final_analysis = {
        "summary": ai_result.get("summary", ""),
        "pros": pros,
        "cons": cons,
        "neutral_points": neutral_points,
        "out_of_scope": False,
    }

    cache_manager.set(cache_key, {
        "analysis": {
            "summary": final_analysis["summary"],
            "pros": [p.model_dump() for p in pros],
            "cons": [c.model_dump() for c in cons],
            "neutral_points": neutral_points,
        },
        "sentiment": sentiment,
    })

    return final_analysis, sentiment, [], False, warnings, domain


def calculate_score_and_confidence(sentiment: dict) -> tuple[float, float]:
    total = sentiment.get("total", 1)
    if total == 0:
        return 0.0, 0.0

    pos = sentiment.get("positive", 0) / 100
    neg = sentiment.get("negative", 0) / 100
    neu = sentiment.get("neutral", 0) / 100

    score = round(max(1.0, min(5.0, (pos - neg + 1) * 2.5)), 2)

    dominant = max(pos, neg, neu)
    sample_factor = min(math.sqrt(total / 8), 1.0)
    
    sentiment_confidence = dominant * (0.25 + 0.75 * sample_factor) * 100
    avg_vader_confidence = sentiment.get("avg_confidence", 0.5) * 100
    
    confidence = (sentiment_confidence * 0.7) + (avg_vader_confidence * 0.3)

    return score, min(100.0, round(confidence, 2))


# ==============================================================================
# SHARED HANDLER
# ==============================================================================
async def _handle_analyze(
    request: Request,
    payload: RawAnalyzeRequest,
    x_api_key: Optional[str],
    endpoint: str,
) -> AnalyzeResponse:
    start_time = time.time()
    request_tracker.active_http += 1

    try:
        api_key = await verify_api_key(x_api_key)
        user_id = hashlib.sha256(api_key.encode()).hexdigest()[:6]

        is_limited, reason = scalable_limiter.is_rate_limited(user_id)
        if is_limited:
            REQUEST_COUNT.labels(endpoint=endpoint, status="rate_limited").inc()
            raise HTTPException(status_code=429, detail=reason)

        if len(payload.raw_text) > MAX_INPUT_SIZE:
            raise HTTPException(status_code=400, detail=f"Input too large. Max {MAX_INPUT_SIZE} chars.")

        reviews = parse_raw_input(payload.raw_text)

        if not reviews:
            raise HTTPException(status_code=400, detail="Could not extract reviews.")

        if len(reviews) > 50:
            raise HTTPException(status_code=400, detail="Too many reviews. Maximum 50 per request.")

        # 🚨 FIX #1 & #2: STRICTER PRE-VALIDATION (threshold 0.4, no partial)
        is_relevant, relevance_score, _ = is_product_review_related(payload.raw_text)
        if not is_relevant:
            category = detect_out_of_scope_category(payload.raw_text)
            REQUEST_COUNT.labels(endpoint=endpoint, status="out_of_scope").inc()
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "OUT_OF_SCOPE",
                    "message": f"Input does not appear to be a product review.",
                    "detected_category": category,
                    "relevance_score": round(relevance_score, 2),
                    "suggestion": "Please provide actual product reviews with ratings, pros/cons, or product feedback."
                }
            )

        scalable_limiter.record_request(user_id)

        detailed = request.query_params.get("detailed", "false").lower() == "true"
        user_focus = payload.user_focus

        try:
            result = await asyncio.wait_for(
                process_analysis_task(reviews, detailed, user_focus),
                timeout=REQUEST_TIMEOUT
            )
            analysis, sentiment, feature_scores, from_cache, ai_warnings, domain = result
        except asyncio.TimeoutError:
            REQUEST_COUNT.labels(endpoint=endpoint, status="timeout").inc()
            return AnalyzeResponse(
                summary="Request timed out – please try again later.",
                pros=[],
                cons=[],
                neutral_points=[],
                sentiment=SentimentBreakdown(positive=0.0, neutral=0.0, negative=0.0, total=0),
                score=0.0,
                confidence=0.0,
                cached=False,
                warnings=[WarningDetail(type="REQUEST_TIMEOUT", message="Processing took too long.")],
                domain=detect_domain(payload.raw_text),
                out_of_scope=False,
            )

        # 🚨 CHECK IF OUT-OF-SCOPE DETECTED DURING PROCESSING
        if analysis.get("out_of_scope"):
            REQUEST_COUNT.labels(endpoint=endpoint, status="out_of_scope").inc()
            return AnalyzeResponse(
                summary="",
                pros=[],
                cons=[],
                neutral_points=[],
                sentiment=SentimentBreakdown(positive=0.0, neutral=0.0, negative=0.0, total=0),
                score=0.0,
                confidence=0.0,
                cached=False,
                warnings=ai_warnings,
                domain=domain,
                out_of_scope=True,
            )

        score, confidence = calculate_score_and_confidence(sentiment)

        pros = analysis["pros"]
        cons = analysis["cons"]

        response = AnalyzeResponse(
            summary=analysis["summary"],
            pros=pros,
            cons=cons,
            neutral_points=analysis.get("neutral_points", []),
            sentiment=SentimentBreakdown(
                positive=sentiment["positive"],
                neutral=sentiment["neutral"],
                negative=sentiment["negative"],
                total=sentiment["total"],
            ),
            score=score,
            confidence=confidence,
            cached=from_cache,
            warnings=(ai_warnings or []),
            domain=domain,
            out_of_scope=False,
        )

        if detailed:
            response.explained_pros = [
                ExplainablePoint(
                    text=p.text,
                    features=p.features,
                    sentiment="positive",
                    polarity_score=round(get_sentiment_polarity(p.text), 3),
                    confidence=round(get_sentiment_confidence(p.text), 3),
                    impact=p.impact,
                )
                for p in pros
            ]
            response.explained_cons = [
                ExplainablePoint(
                    text=c.text,
                    features=c.features,
                    sentiment="negative",
                    polarity_score=round(get_sentiment_polarity(c.text), 3),
                    confidence=round(get_sentiment_confidence(c.text), 3),
                    impact=c.impact,
                )
                for c in cons
            ]
            response.feature_scores = feature_scores

        REQUEST_COUNT.labels(endpoint=endpoint, status="success").inc()
        return response

    except HTTPException:
        raise
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, status="error").inc()
        logger.error("ANALYTICS ERROR", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        request_tracker.active_http -= 1


# ==============================================================================
# API ROUTES
# ==============================================================================
@app.post(f"{API_V1_PREFIX}/analyze-raw", response_model=AnalyzeResponse, tags=["v1"])
async def v1_analyze_raw_text(
    request: Request,
    payload: RawAnalyzeRequest,
    x_api_key: Optional[str] = Header(None),
) -> AnalyzeResponse:
    return await _handle_analyze(request, payload, x_api_key, f"{API_V1_PREFIX}/analyze-raw")


@app.post(f"{API_V1_PREFIX}/analyze", response_model=AnalyzeResponse, tags=["v1"])
async def v1_analyze_reviews(
    request: Request,
    payload: ReviewRequest,
    x_api_key: Optional[str] = Header(None),
) -> AnalyzeResponse:
    raw_text = "\n".join(payload.reviews)
    return await _handle_analyze(
        request, RawAnalyzeRequest(raw_text=raw_text), x_api_key, f"{API_V1_PREFIX}/analyze"
    )


@app.post(f"{API_V2_PREFIX}/analyze-raw", response_model=AnalyzeResponse, tags=["v2"])
async def v2_analyze_raw_text(
    request: Request,
    payload: RawAnalyzeRequest,
    x_api_key: Optional[str] = Header(None),
) -> AnalyzeResponse:
    return await _handle_analyze(request, payload, x_api_key, f"{API_V2_PREFIX}/analyze-raw")


@app.post(f"{API_V2_PREFIX}/analyze", response_model=AnalyzeResponse, tags=["v2"])
async def v2_analyze_reviews(
    request: Request,
    payload: ReviewRequest,
    x_api_key: Optional[str] = Header(None),
) -> AnalyzeResponse:
    raw_text = "\n".join(payload.reviews)
    return await _handle_analyze(
        request, RawAnalyzeRequest(raw_text=raw_text), x_api_key, f"{API_V2_PREFIX}/analyze"
    )


@app.post("/analyze-raw", response_model=AnalyzeResponse, tags=["legacy"])
async def analyze_raw_text(
    request: Request,
    payload: RawAnalyzeRequest,
    x_api_key: Optional[str] = Header(None),
) -> AnalyzeResponse:
    return await _handle_analyze(request, payload, x_api_key, "/analyze-raw")


@app.post("/analyze", response_model=AnalyzeResponse, tags=["legacy"])
async def analyze_reviews(
    request: Request,
    payload: ReviewRequest,
    x_api_key: Optional[str] = Header(None),
) -> AnalyzeResponse:
    raw_text = "\n".join(payload.reviews)
    return await _handle_analyze(
        request, RawAnalyzeRequest(raw_text=raw_text), x_api_key, "/analyze"
    )


@app.post("/analyze-stream", tags=["streaming"])
async def analyze_stream(
    request: Request,
    payload: RawAnalyzeRequest,
    x_api_key: Optional[str] = Header(None),
):
    await verify_api_key(x_api_key)
    
    if len(payload.raw_text) > MAX_INPUT_SIZE:
        raise HTTPException(status_code=400, detail=f"Input too large. Max {MAX_INPUT_SIZE} chars.")
    
    # 🚨 FIX #1 & #2: STRICTER PRE-VALIDATION for streaming too
    is_relevant, relevance_score, _ = is_product_review_related(payload.raw_text)
    if not is_relevant:
        category = detect_out_of_scope_category(payload.raw_text)
        raise HTTPException(
            status_code=400,
            detail={
                "error": "OUT_OF_SCOPE",
                "message": "Input does not appear to be a product review.",
                "detected_category": category,
                "relevance_score": round(relevance_score, 2),
            }
        )
    
    reviews = parse_raw_input(payload.raw_text)
    
    if not reviews:
        raise HTTPException(status_code=400, detail="Could not extract reviews.")
    
    if len(reviews) > 50:
        raise HTTPException(status_code=400, detail="Too many reviews. Maximum 50 per request.")
    
    async def stream_results():
        raw_text = " ".join(reviews)
        detected_domain = detect_domain(raw_text)
        yield json.dumps({"type": "domain", "data": detected_domain}) + "\n"
        await asyncio.sleep(STREAM_CHUNK_DELAY)
        
        model_name = resolve_model_name(os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
        ai_result = await gemini_manager.analyze_reviews(reviews, model_name, detected_domain)
        
        if ai_result and ai_result.get("out_of_scope"):
            yield json.dumps({
                "type": "error",
                "data": "OUT_OF_SCOPE",
                "reason": ai_result.get("reason", "Not a product review"),
                "category": ai_result.get("detected_category", "unknown")
            }) + "\n"
        elif ai_result:
            yield json.dumps({"type": "summary", "data": ai_result.get("summary", "")}) + "\n"
            await asyncio.sleep(STREAM_CHUNK_DELAY)
            
            yield json.dumps({"type": "pros", "data": ai_result.get("pros", [])}) + "\n"
            await asyncio.sleep(STREAM_CHUNK_DELAY)
            
            yield json.dumps({"type": "cons", "data": ai_result.get("cons", [])}) + "\n"
            await asyncio.sleep(STREAM_CHUNK_DELAY)
            
            yield json.dumps({"type": "sentiment", "data": ai_result.get("ai_sentiment", {})}) + "\n"
        else:
            yield json.dumps({"type": "error", "data": "AI analysis failed"}) + "\n"
        
        yield "[DONE]\n"
    
    return StreamingResponse(
        stream_results(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        }
    )
    


# ==============================================================================
# ADMIN ENDPOINTS
# ==============================================================================
@app.get("/stats")
async def get_stats(x_api_key: Optional[str] = Header(None)):
    await verify_api_key(x_api_key)
    return {
        "cache": cache_manager.lru_cache.get_stats(),
        "polarity_cache": {
            "size": get_sentiment_polarity_cached.cache_info().currsize,
            "hits": get_sentiment_polarity_cached.cache_info().hits,
        },
        "feature_cache": {
            "size": extract_feature_cached.cache_info().currsize,
            "hits": extract_feature_cached.cache_info().hits,
        }
    }


@app.post("/cache/clear")
async def clear_cache(x_api_key: Optional[str] = Header(None)):
    await verify_api_key(x_api_key)
    cache_manager.lru_cache.cache.clear()
    get_sentiment_polarity_cached.cache_clear()
    extract_feature_cached.cache_clear()
    return {"status": "cache cleared"}


@app.get("/admin/gemini-health")
async def get_gemini_health(x_api_key: Optional[str] = Header(None)):
    await verify_api_key(x_api_key)
    return gemini_manager.get_health()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "message": "AI Product Review Aggregator API v25.0-ai-validated",
        "docs": "/docs",
        "v1": f"{API_V1_PREFIX}/analyze-raw",
        "v2": f"{API_V2_PREFIX}/analyze-raw",
        "streaming": "/analyze-stream",
    }


# ==============================================================================
# TESTS - UPDATED WITH ALL FIXES
# ==============================================================================
def _run_tests():
    import traceback
    results = []

    def test(name, fn):
        try:
            fn()
            results.append(("PASS", name))
        except AssertionError as e:
            results.append(("FAIL", f"{name}: {e}"))
        except Exception as e:
            results.append(("ERROR", f"{name}: {traceback.format_exc(limit=1)}"))

    def check(condition: bool, message: str):
        if not condition:
            raise AssertionError(message)

    # Test 1: Out-of-scope detection - sports
    is_relevant, score, detected_type = is_product_review_related("Tell me about cricket match")
    check(not is_relevant, f"Cricket query should be rejected. Score: {score}")
    
    # Test 2: Out-of-scope detection - finance
    is_relevant, score, detected_type = is_product_review_related("Is Bitcoin going up?")
    check(not is_relevant, f"Finance query should be rejected. Score: {score}")
    
    # Test 3: Product review should be accepted
    is_relevant, score, detected_type = is_product_review_related("Great phone, battery lasts all day. Camera is amazing but screen scratches easily.")
    check(is_relevant, f"Product review should be accepted. Score: {score}")
    
    # Test 4: Out-of-scope detection - politics
    is_relevant, score, detected_type = is_product_review_related("Who won the election?")
    check(not is_relevant, f"Politics query should be rejected. Score: {score}")
    
    # Test 5: Domain detection
    text = "Nice fabric but color faded quickly after washing"
    domain = detect_domain(text)
    check(domain == "clothing", f"Should detect clothing domain")

    # Test 6: Feature extraction
    features = extract_features_with_context("Camera is good but battery drains fast", "electronics")
    check(len(features) >= 2, f"Should detect multiple features, got {features}")
    
    # Test 7: Relevance score calculation
    score = calculate_relevance_score("I bought a laptop last week. The battery life is great but it runs hot.")
    check(score > 0.4, f"Product review should have high score (>=0.4), got {score}")
    
    # Test 8: Category detection
    category = detect_out_of_scope_category("Tell me about the football match")
    check(category == "sports", f"Should detect sports category, got {category}")

    # Test 9: FIX #3 - Cricket bat review should be ACCEPTED (has product context)
    is_relevant, score, detected_type = is_product_review_related("Cricket bat review: I bought this bat last month. Great quality and balance. Perfect for practice sessions.")
    check(is_relevant, f"Cricket bat review WITH product context should be ACCEPTED. Score: {score}, Type: {detected_type}")
    check(score >= 0.4, f"Score should be >=0.4, got {score}")
    
    # Test 10: FIX #3 - Football score prediction should be REJECTED
    is_relevant, score, detected_type = is_product_review_related("What's your football score prediction for tomorrow's match?")
    check(not is_relevant, f"Football score prediction should be REJECTED. Score: {score}")
    
    # Test 11: FIX #3 - Movie review (entertainment) should be REJECTED
    is_relevant, score, detected_type = is_product_review_related("I watched the new movie on Netflix last night. The acting was great but the plot was confusing.")
    check(not is_relevant, f"Movie review should be REJECTED. Score: {score}")

    # Test 12: FIX #2 - "Partial" classification should NOT be accepted
    is_relevant, score, detected_type = is_product_review_related("The battery is okay but the camera could be better. I use it daily.")
    # This should either be accepted (>=0.4) or rejected (<0.4) - but NEVER "partial"
    if is_relevant:
        check(detected_type == "product_review", f"If relevant, must be 'product_review', not 'partial'. Got: {detected_type}")
    else:
        check(detected_type == "irrelevant", f"If not relevant, must be 'irrelevant', not 'partial'. Got: {detected_type}")
    
    # Test 13: FIX #1 - Stricter threshold (0.4 instead of 0.3)
    # A borderline case should now be rejected
    is_relevant, score, detected_type = is_product_review_related("Battery lasts long")
    check(not is_relevant or score >= 0.4, f"Borderline case with new threshold (0.4) should have score >=0.4 if accepted. Got: {score}")

    passed = sum(1 for r in results if r[0] == "PASS")
    failed = sum(1 for r in results if r[0] != "PASS")
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 50)
    for status, name in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name}")


# ==============================================================================
# STARTUP
# ==============================================================================
@app.on_event("startup")
async def startup_event():
    _build_automaton_maps()
    _build_sentiment_automaton()
    logger.info("API v25.0-ai-validated started with STRICTER OUT-OF-SCOPE DETECTION (fixes applied)")

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.on_event("shutdown")
async def shutdown_event():
    await gemini_manager.close()
    logger.info("API shutdown - cleaned up resources")


# ==============================================================================
# ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _run_tests()
        sys.exit(0)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
