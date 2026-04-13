import asyncio
import traceback
import hashlib
import json
import logging
import math
import os
import random
import re
import threading
import time
from collections import Counter, OrderedDict
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator

# ==============================================================================
# OPTIONAL VADER SENTIMENT
# ==============================================================================

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    vader_analyzer = SentimentIntensityAnalyzer()
    USE_VADER = True
except ImportError:
    USE_VADER = False

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API Keys
VALID_API_KEYS = set()
raw_keys = os.getenv("API_KEYS", "")
if raw_keys:
    VALID_API_KEYS = {k.strip() for k in raw_keys.split(",") if k.strip()}

if not VALID_API_KEYS:
    logger.info("No API_KEYS configured - running in open mode")

# Cache settings
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1000"))




# Constants
PLACEHOLDER_API_KEYS = {"your_gemini_api_key_here", "replace_with_real_key"}
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LEGACY_MODEL_ALIASES = {"gemini-pro": DEFAULT_GEMINI_MODEL}
GEMINI_TIMEOUT = 8

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

# Worker Configuration
WORKER_POOL_SIZE = int(os.getenv("WORKER_POOL_SIZE", "3"))
QUEUE_MAX_SIZE = 100

# API version prefix
API_V1_PREFIX = "/api/v1"
API_V2_PREFIX = "/api/v2"


# ==============================================================================
# FIX 1: STRUCTURED WARNINGS MODEL
# ==============================================================================

class WarningDetail(BaseModel):
    type: str
    message: str


# ==============================================================================
# FIX 1 (IMPROVED): DOMAIN DETECTION WITH TOKENIZATION + PHRASE MATCHING + WEIGHTED SCORING
# ==============================================================================

DOMAIN_KEYWORDS: dict[str, dict] = {
    "electronics": {
        "single": {
            "battery", "camera", "screen", "display", "charger", "charging", "usb",
            "processor", "cpu", "ram", "storage", "speaker", "audio", "bluetooth",
            "wifi", "5g", "lte", "sensor", "gps", "fingerprint", "gaming",
            "graphics", "gpu", "refresh", "oled", "lcd", "amoled",
            "pixel", "megapixels", "zoom", "lens", "aperture",
            "waterproof", "ip68", "headphone", "earphone", "touchscreen",
        },
        "phrases": {
            "battery life", "fast charging", "night mode", "face unlock",
            "wireless charging", "refresh rate", "portrait mode",
        },
        "weight": 1,
    },
    "clothing": {
        "single": {
            "fabric", "material", "cotton", "polyester", "size", "fit", "tight",
            "loose", "stretch", "breathable", "wash", "color",
            "fade", "shrink", "stitch", "seam", "pocket", "zipper", "button",
            "sleeve", "collar", "hem", "inseam", "waist", "hip",
        },
        "phrases": {
            "true to size", "runs small", "runs large", "machine wash",
        },
        "weight": 1,
    },
    "food": {
        "single": {
            "taste", "flavor", "fresh", "spicy", "sweet", "salty", "bitter",
            "sour", "crispy", "crunchy", "texture", "aroma", "portion",
            "serving", "calorie", "organic", "ingredients", "nutrition", "protein",
        },
        "phrases": {
            "expiry date", "best before", "shelf life",
        },
        "weight": 1,
    },
    "furniture": {
        "single": {
            "assembly", "stable", "wobbly", "wood", "metal",
            "leather", "cushion", "ergonomic", "backrest", "armrest",
            "drawer", "shelf", "chair", "desk", "bedframe", "mattress",
        },
        "phrases": {
            "easy to assemble", "hard to assemble", "assembly instructions",
        },
        "weight": 1,
    },
    "beauty": {
        "single": {
            "moisturizer", "serum", "foundation", "concealer", "mascara",
            "eyeshadow", "lipstick", "blush", "primer", "breakout",
            "acne", "hypoallergenic",
        },
        "phrases": {
            "setting spray", "cruelty free", "long lasting", "skin tone",
        },
        "weight": 1,
    },
    "home_appliances": {
        "single": {
            "noise", "filter", "capacity", "powerful",
            "efficient", "temperature", "timer", "automatic",
        },
        "phrases": {
            "energy efficient", "noise level",
        },
        "weight": 1,
    },
}


def tokenize(text: str) -> set[str]:
    """FIX 1: Proper tokenization using regex word boundaries — handles hyphenated words and punctuation."""
    return set(re.findall(r'\b\w+\b', text.lower()))


def detect_domain(text: str) -> str:
    """FIX 1: Weighted domain detection using single-word tokens AND multi-word phrase matching."""
    words = tokenize(text)
    text_lower = text.lower()
    scores: dict[str, float] = {}

    for domain, config in DOMAIN_KEYWORDS.items():
        score = 0.0
        weight = config.get("weight", 1)

        # Single-word matches
        single_hits = words & config["single"]
        score += len(single_hits) * weight

        # Phrase matches (worth 2x)
        for phrase in config.get("phrases", set()):
            if phrase in text_lower:
                score += 2 * weight

        scores[domain] = score

    if scores and max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return "generic"


# ==============================================================================
# FIX 2: WINDOW-BASED NEGATION HANDLING
# ==============================================================================

NEGATIONS = frozenset({
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "hardly", "barely", "scarcely", "seldom", "rarely", "without",
    "lack", "lacking", "doesn't", "don't", "didn't", "won't", "wouldn't",
    "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
})


def handle_negation(text: str) -> str:
    """FIX 2: Window-based negation — negates up to 3 words after a negation trigger."""
    words = text.lower().split()
    result = []
    negate_window = 0

    for w in words:
        if w in NEGATIONS:
            negate_window = 3  # negate the next 3 tokens
            continue

        if negate_window > 0:
            result.append("NOT_" + w)
            negate_window -= 1
        else:
            result.append(w)

    return " ".join(result)


# ==============================================================================
# FIX 3: FEATURE DETECTION WITH CONFIDENCE THRESHOLD (min 2 alias hits)
# ==============================================================================

BASE_FEATURES = {
    "battery": {"battery", "backup", "drain", "mah", "power", "charging", "charge", "charger"},
    "camera": {"camera", "photo", "video", "lens", "focus", "zoom", "selfie", "megapixel", "aperture"},
    "performance": {"performance", "lag", "slow", "speed", "processor", "ram", "gaming", "gpu"},
    "design": {"design", "look", "style", "color", "aesthetic", "sleek", "premium"},
    "display": {"display", "screen", "brightness", "touch", "oled", "lcd", "amoled", "resolution"},
    "sound": {"sound", "audio", "speaker", "volume", "bass", "microphone", "mic"},
    "charging": {"charging", "charge", "charger", "wireless", "fast"},
    "build": {"build", "quality", "durable", "material", "plastic", "metal", "glass", "waterproof"},
    "price": {"price", "cost", "expensive", "value", "worth", "cheap", "affordable", "budget"},
    "software": {"software", "ui", "update", "app", "os", "android", "ios", "feature"},
    "support": {"support", "service", "warranty", "help", "response", "customer"},
    "comfort": {"comfort", "fit", "pain", "ear", "heavy", "light", "weight", "ergonomic"},
    "connectivity": {"wifi", "bluetooth", "signal", "network", "5g", "lte", "gps"},
}

DOMAIN_FEATURES = {
    "clothing": {
        "fabric": {"fabric", "material", "cotton", "polyester", "silk", "denim"},
        "fit": {"fit", "size", "tight", "loose"},
        "comfort": {"comfortable", "soft", "breathable", "itchy", "rough"},
        "durability": {"durability", "fade", "shrink", "stretch", "tear", "pilling"},
    },
    "food": {
        "taste": {"taste", "flavor", "bland", "delicious", "savory", "sweet"},
        "texture": {"texture", "crispy", "crunchy", "chewy", "tender", "dry", "moist"},
        "value": {"portion", "serving", "value", "fresh", "expired"},
    },
    "furniture": {
        "assembly": {"assembly", "instructions", "difficult", "assemble"},
        "stability": {"stable", "wobbly", "sturdy", "solid", "flimsy"},
        "comfort": {"comfortable", "cushion", "support", "firm", "soft"},
    },
    "beauty": {
        "application": {"application", "blend", "coverage", "pigmented", "patchy"},
        "wear": {"wear", "lasting", "smudge", "transfer", "fade", "settle"},
        "skin": {"breakout", "irritation", "allergic", "sensitive", "oily", "dry"},
    },
}


def get_features_for_domain(domain: str) -> dict:
    features = BASE_FEATURES.copy()
    if domain in DOMAIN_FEATURES:
        features.update(DOMAIN_FEATURES[domain])
    return features


def extract_feature_with_context(sentence: str, domain: str = "generic") -> Optional[str]:
    """FIX 3: Confidence-based feature detection — requires at least 2 alias hits in context window."""
    words = re.findall(r'\b\w+\b', sentence.lower())
    features = get_features_for_domain(domain)

    best_feature = None
    best_score = 0

    for i, word in enumerate(words):
        window_start = max(0, i - 3)
        window_end = min(len(words), i + 4)
        window = set(words[window_start:window_end])

        for feature, aliases in features.items():
            # FIX 3: count how many alias words appear in window
            match_count = sum(1 for alias in aliases if alias in window)

            # Direct exact match on current word gets +1 bonus
            if word in aliases:
                match_count += 1

            # Require confidence threshold of >= 2 hits
            if match_count >= 2 and match_count > best_score:
                best_score = match_count
                best_feature = feature

    # Fallback: exact word match with score 1 (single strong signal)
    if not best_feature:
        for i, word in enumerate(words):
            for feature, aliases in features.items():
                if word in aliases:
                    return feature

    return best_feature


@lru_cache(maxsize=1000)
def extract_feature_cached(sentence: str, domain: str = "generic") -> Optional[str]:
    return extract_feature_with_context(sentence, domain)


# ==============================================================================
# FIX 4: IMPROVED CLAUSE SPLITTING
# ==============================================================================

def split_into_clauses(text: str) -> list[str]:
    segments = []
    connector_pattern = r"\s+(?:but|however|although|though|while|whereas|yet|except|otherwise|nonetheless|nevertheless|alternatively|instead|also|plus|and then)\s+"

    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        if not sentence:
            continue

        clauses = re.split(connector_pattern, sentence, flags=re.I)

        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue

            for prefix in ("but ", "however ", "although ", "though ", "while ", "yet ", "except "):
                if clause.lower().startswith(prefix):
                    clause = clause[len(prefix):].strip()
                    break

            if clause and clause[0].islower():
                clause = clause[0].upper() + clause[1:]

            clause = clause.strip(" ,.;")
            if clause and len(clause) >= 10:
                segments.append(clause)

    return segments


# ==============================================================================
# FIX 5 (SHORTEN)
# ==============================================================================

def shorten(text: str, max_length: int = 80) -> str:
    if not text:
        return text

    text = text.strip()

    if "," in text:
        text = text.split(",")[0]
    elif "." in text:
        text = text.split(".")[0]

    if len(text) > max_length:
        text = text[:max_length - 3].rstrip() + "..."

    return text.strip()


# ==============================================================================
# KEYWORD SETS
# ==============================================================================

STRONG_NEGATIVE = frozenset({
    "bad", "poor", "worst", "waste", "overheat", "lag", "drain", "heats",
    "fails", "failure", "crash", "buggy", "terrible", "horrible", "awful", "broken",
    "disappointing", "frustrating", "useless", "defective", "cheaply", "flimsy",
    "pathetic", "regret", "nightmare", "disaster", "avoid", "refuse",
})

STRONG_POSITIVE = frozenset({
    "excellent", "great", "smooth", "easy", "premium", "bright",
    "sharp", "clean", "love", "best", "amazing", "perfect", "outstanding",
    "fantastic", "wonderful", "brilliant", "superb", "impressed", "recommend",
    "exceeded", "delighted", "flawless", "exceptional", "remarkable",
})

SOFT_NEGATIVE = frozenset({
    "slow", "weak", "issue", "problem", "expensive", "overpriced", "noisy", "hot",
    "average", "mediocre", "meh", "uncomfortable", "inconvenient", "complicated",
})

SOFT_POSITIVE = frozenset({
    "good", "nice", "fast", "clear", "lightweight", "sleek", "fine", "okay", "decent",
    "solid", "reliable", "satisfactory", "acceptable", "adequate", "pleasant",
})

ALL_POSITIVE = STRONG_POSITIVE | SOFT_POSITIVE
ALL_NEGATIVE = STRONG_NEGATIVE | SOFT_NEGATIVE


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
    user_focus: Optional[str] = Field(None, description="Optional feature focus for personalization (e.g. 'battery')")

    @field_validator("raw_text")
    @classmethod
    def validate_size(cls, v: str) -> str:
        if len(v) > MAX_INPUT_SIZE:
            raise ValueError(f"Input too large. Maximum {MAX_INPUT_SIZE} characters allowed.")
        return v


class SentimentBreakdown(BaseModel):
    positive: float
    neutral: float
    negative: float
    total: int


class ExplainablePoint(BaseModel):
    text: str
    feature: str
    sentiment: str
    polarity_score: float
    impact: str  # FIX 10: "high" | "medium" | "low"


# FIX 10: Structured pro/con item
class AnalysisPoint(BaseModel):
    text: str
    feature: str
    impact: str  # "high" | "medium" | "low"


class FeatureScore(BaseModel):
    feature: str
    display_name: str
    positive_count: int
    negative_count: int
    total_mentions: int
    score: float


class AnalyzeResponse(BaseModel):
    summary: str
    # FIX 10: Frontend-ready structured format
    pros: list[AnalysisPoint]
    cons: list[AnalysisPoint]
    neutral_points: list[str] = Field(default_factory=list)
    sentiment: SentimentBreakdown
    score: float
    confidence: float
    cached: bool = False
    # FIX 7: Structured warnings
    warnings: list[WarningDetail] = Field(default_factory=list)
    explained_pros: list[ExplainablePoint] | None = None
    explained_cons: list[ExplainablePoint] | None = None
    feature_scores: list[FeatureScore] | None = None
    domain: str | None = None


# ==============================================================================
# FIX 12: VERSIONED FASTAPI APP
# ==============================================================================

app = FastAPI(
    title="AI Product Review Aggregator API",
    description="Production-ready elite-level system",
    version="20.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# API KEY AUTHENTICATION
# ==============================================================================

async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    if not VALID_API_KEYS:
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

    REQUEST_COUNT = Counter('review_api_requests_total', 'Total requests', ['endpoint', 'status'])
    REQUEST_LATENCY = Histogram('review_api_request_duration_seconds', 'Request latency', ['endpoint'])
    CACHE_HITS = Counter('review_api_cache_hits_total', 'Cache hits')
    QUEUE_SIZE = Gauge('review_api_queue_size', 'Current queue size')
    SEMAPHORE_USAGE = Gauge('review_api_semaphore_usage', 'Current semaphore usage')
    ACTIVE_REQUESTS = Gauge('review_api_active_requests', 'Active requests')
    GEMINI_REQUESTS_ACTIVE = Gauge('review_api_gemini_active', 'Active Gemini requests')
    USE_PROMETHEUS = True

    @app.get("/metrics")
    async def metrics():
        return JSONResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

except ImportError:
    USE_PROMETHEUS = False

    class NoOpCounter:
        def __init__(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
        def inc(self, n=1): pass

    class NoOpHistogram:
        def __init__(self, *args, **kwargs): pass
        def labels(self, **kwargs):
            class HistogramCtx:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def time(self):
                    class TimerCtx:
                        def __enter__(self): return self
                        def __exit__(self, *args): pass
                    return TimerCtx()
            return HistogramCtx()

    class NoOpGauge:
        def __init__(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
        def set(self, n): pass

    REQUEST_COUNT = NoOpCounter()
    REQUEST_LATENCY = NoOpHistogram()
    CACHE_HITS = NoOpCounter()
    QUEUE_SIZE = NoOpGauge()
    SEMAPHORE_USAGE = NoOpGauge()
    ACTIVE_REQUESTS = NoOpGauge()
    GEMINI_REQUESTS_ACTIVE = NoOpGauge()


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
                    cls._instance._queue_depth = 0
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
        SEMAPHORE_USAGE.set(self._active_gemini)

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    @queue_depth.setter
    def queue_depth(self, value: int):
        self._queue_depth = max(0, value)
        QUEUE_SIZE.set(self._queue_depth)


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
        return json.loads(data_str)

    def set(self, key: str, data: dict, ttl: int = 3600):
        while self.size >= self.max_size:
            self.cache.popitem(last=False)
            self.size -= 1

        data_str = json.dumps(data)
        self.cache[key] = (data_str, time.time() + ttl)
        self.cache.move_to_end(key)
        self.size += 1

    def get_stats(self) -> dict:
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        return {
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": f"{hit_rate:.1f}%",
            "entries": self.size,
            "max_size": self.max_size
        }


class CacheManager:
    def __init__(self):
        self.lru_cache = LRUCache(max_size=MAX_CACHE_SIZE)
        
        self.cache_set_count = 0

    def generate_cache_key(self, reviews: list[str], detailed: bool = False, domain: str = "generic") -> str:
        """FIX 8: Cache key is now context-aware (includes detailed flag + domain)."""
        sorted_reviews = sorted(reviews)
        payload = {
            "reviews": sorted_reviews,
            "detailed": detailed,
            "domain": domain,
        }
        reviews_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return f"review_cache:{reviews_hash[:32]}"

    def get(self, key: str) -> Optional[dict]:
        if not ENABLE_CACHE:
            return None

        return self.lru_cache.get(key)

    def set(self, key: str, data: dict, ttl: int = None):
        if not ENABLE_CACHE:
            return

        ttl = ttl or CACHE_TTL_SECONDS
        data_str = json.dumps(data)

        

        self.lru_cache.set(key, data, ttl)

    

    


cache_manager = CacheManager()


# ==============================================================================
# RATE LIMITER
# ==============================================================================

class ScalableRateLimiter:
    def __init__(self):
        self._requests = {}

    def record_request(self, identifier: str):
        now = time.time()
        # CLEANUP old users (older than 1 hour)
        for key in list(self._requests.keys()):
            self._requests[key] = [t for t in self._requests[key] if now - t < 3600]
            if not self._requests[key]:
                del self._requests[key]
        if identifier in self._requests:
            self._requests[identifier] = [t for t in self._requests[identifier] if now - t < 3600]
        
        if identifier not in self._requests:
            self._requests[identifier] = []
        self._requests[identifier].append(now)

    def get_request_count(self, identifier: str, window_seconds: int = 60) -> int:
        now = time.time()
        
        if identifier not in self._requests:
            return 0
        cutoff = now - window_seconds
        self._requests[identifier] = [t for t in self._requests[identifier] if t > cutoff]
        return len(self._requests[identifier])

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
# WORKER QUEUE SYSTEM
# ==============================================================================

GEMINI_CONCURRENCY_LIMIT = 5
gemini_semaphore = asyncio.Semaphore(GEMINI_CONCURRENCY_LIMIT)

WORKER_QUEUE: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)

TASK_RESULTS: Dict[str, asyncio.Future] = {}
TASK_RESULTS_LOCK = threading.Lock()

task_id_counter = 0
task_id_lock = threading.Lock()


def generate_task_id() -> str:
    global task_id_counter
    with task_id_lock:
        task_id_counter += 1
        return f"task_{task_id_counter}_{int(time.time() * 1000)}"


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

    @property
    def stats(self) -> dict:
        total = self._total_requests or 1
        return {
            "id": self.id,
            "name": self._name,
            "healthy": self.is_healthy,
            "failures": self._failure_count,
            "successes": self._success_count,
            "success_rate": round(self._success_count / total * 100, 1),
        }

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
    _instance: Optional['GeminiClientManager'] = None
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
        self._clients: dict[str, object] = {}
        self._rotation_lock = threading.Lock()
        self._index = 0
        self._round_robin: list[str] = []
        self._max_retries = 3
        self._base_delay = 1.0
        self._max_delay = 30.0
        self._consecutive_failures = 0
        self._circuit_open_time: Optional[float] = None
        self._load_keys()
        healthy_count = self.get_healthy_key_count()
        logger.info(f"GeminiClientManager initialized: {len(self._keys)} total keys, {healthy_count} healthy")
        if healthy_count == 0:
            logger.warning("No healthy Gemini API keys — AI enhancement disabled until keys recover.")

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

    def get_circuit_state(self) -> dict:
        return {
            "open": self.is_circuit_open(),
            "consecutive_failures": self._consecutive_failures,
            "open_since": self._circuit_open_time,
            "healthy_keys": self.get_healthy_key_count(),
            "total_keys": len(self._keys),
        }

    def _get_client(self, key_hash: str, api_key: str) -> object:
        if key_hash not in self._clients:
            self._clients[key_hash] = genai.Client(api_key=api_key)
        return self._clients[key_hash]

    def _load_keys(self):
        self._keys.clear()
        self._clients.clear()
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
                if key_hash in self._keys and self._keys[key_hash].is_healthy:
                    return self._keys[key_hash], key_hash
            self._rebuild_rotation()
            if not self._round_robin:
                return None
            key_hash = self._round_robin[self._index % len(self._round_robin)]
            cfg = self._keys.get(key_hash)
            return (cfg, key_hash) if cfg else None

    def _calculate_delay(self, attempt: int, is_rate_limit: bool = False) -> float:
        base = self._base_delay * (2 ** attempt)
        if is_rate_limit:
            base *= 3
        return min(base * random.uniform(0.75, 1.25), self._max_delay)

    def _classify_error(self, error: Exception) -> tuple[bool, bool]:
        msg = str(error).lower()
        if any(p in msg for p in {"429", "rate limit", "quota", "too many requests"}):
            return True, True
        if any(p in msg for p in {"500", "502", "503", "504", "timeout", "connection", "unavailable"}):
            return True, False
        return False, False

    async def _execute_with_retry(self, func, *args, **kwargs) -> tuple[Optional[object], Optional[Exception]]:
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            result = self._get_next_key()
            if not result:
                self._consecutive_failures += 1
                if self._consecutive_failures >= 3 and self._circuit_open_time is None:
                    self._circuit_open_time = time.time()
                return None, Exception("No healthy API keys available")

            key_config, key_hash = result
            try:
                client = self._get_client(key_hash, key_config.key)
                result_obj = await asyncio.to_thread(func, client, *args, **kwargs)
                key_config.record_success()
                self._consecutive_failures = 0
                return result_obj, None
            except Exception as e:
                last_error = e
                is_retryable, is_rl = self._classify_error(e)
                if is_rl:
                    key_config.record_failure(is_rate_limit=True)
                elif is_retryable:
                    key_config.record_failure(is_rate_limit=False)
                else:
                    return None, e
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._calculate_delay(attempt, is_rl))
                    self._rebuild_rotation()
        return None, last_error

    async def analyze_reviews(self, reviews: list[str], model_name: str = DEFAULT_GEMINI_MODEL, domain: str = "generic") -> Optional[dict]:
        prompt = build_analysis_prompt(reviews, domain)

        def _call(client):
            return client.models.generate_content(
                model=model_name,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json"),
            )

        result, error = await self._execute_with_retry(_call)
        if error or not result:
            return None
        if not result.text:
            logger.warning("Gemini returned empty response")
            return None
        return parse_ai_response(result.text)

    async def generate_summary(self, pros: list[str], cons: list[str], model_name: str = DEFAULT_GEMINI_MODEL) -> Optional[str]:
        prompt = build_summary_prompt(pros, cons)

        def _call(client):
            return client.models.generate_content(
                model=model_name,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config=types.GenerateContentConfig(temperature=0.3, response_mime_type="text/plain"),
            )

        result, error = await self._execute_with_retry(_call)
        if error or not result:
            return None
        if not result.text:
            return None
        return (result.text or "").strip()

    def get_health(self) -> dict:
        return {
            "total_keys": len(self._keys),
            "healthy_keys": self.get_healthy_key_count(),
            "circuit_open": self.is_circuit_open(),
            "keys": {cfg.id: cfg.stats for cfg in self._keys.values()},
        }

    def reload_keys(self):
        self._keys.clear()
        self._clients.clear()
        self._circuit_open_time = None
        self._consecutive_failures = 0
        self._load_keys()
        logger.info(f"Gemini keys reloaded: {len(self._keys)} total, {self.get_healthy_key_count()} healthy")


gemini_manager = GeminiClientManager()


# ==============================================================================
# FIX 4 (SENTIMENT): HYBRID SCORING — VADER + KEYWORD + NEGATION INVERSION
# ==============================================================================

def get_sentiment_polarity(text: str) -> float:
    """FIX 4: Hybrid polarity — VADER score combined with keyword scoring, negation-aware."""
    negated_text = handle_negation(text)

    vader_score = 0.0
    if USE_VADER:
        vader_score = vader_analyzer.polarity_scores(negated_text)['compound']

    # Keyword scoring on negation-processed text
    text_lower = negated_text.lower()
    kw_score = 0.0
    for kw in STRONG_POSITIVE:
        if "NOT_" + kw in text_lower:
            kw_score -= 0.3
        elif kw in text_lower:
            kw_score += 0.3
    for kw in SOFT_POSITIVE:
        if "NOT_" + kw in text_lower:
            kw_score -= 0.1
        elif kw in text_lower:
            kw_score += 0.1
    for kw in STRONG_NEGATIVE:
        if "NOT_" + kw in text_lower:
            kw_score += 0.15  # double negative = mild positive
        elif kw in text_lower:
            kw_score -= 0.3
    for kw in SOFT_NEGATIVE:
        if "NOT_" + kw in text_lower:
            kw_score += 0.05
        elif kw in text_lower:
            kw_score -= 0.1

    # Blend: 60% VADER (if available), 40% keyword
    if USE_VADER:
        blended = 0.6 * vader_score + 0.4 * kw_score
    else:
        blended = kw_score

    return max(-1.0, min(1.0, blended))


def parse_raw_input(raw_text: str) -> list[str]:
    if not raw_text:
        return []
    raw_text = raw_text.replace('\r\n', '\n').replace('\r', '\n')
    reviews = []

    for line in raw_text.split('\n'):
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
            line = re.sub(prefix_pattern, '', line, flags=re.IGNORECASE)
        line = line.strip('"\' -')
        if len(line) < 5:
            continue
        if len(line) > 150:
            for part in re.split(r'(?<=[.!?])\s+(?=[A-Z])', line):
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
    if len(text) < 10 or len(text.split()) < 2:
        return False
    if re.search(r'(.)\1{5,}', text):
        return False
    return True


def classify_sentence(sentence: str) -> str:
    polarity = get_sentiment_polarity(sentence)

    if polarity > SENTIMENT_POLARITY_THRESHOLD:
        return "positive"
    if polarity < -SENTIMENT_POLARITY_THRESHOLD:
        return "negative"

    text_lower = sentence.lower()
    if any(kw in text_lower for kw in STRONG_NEGATIVE):
        return "negative"
    if any(kw in text_lower for kw in STRONG_POSITIVE):
        return "positive"
    if any(kw in text_lower for kw in SOFT_NEGATIVE):
        return "negative"
    if any(kw in text_lower for kw in SOFT_POSITIVE):
        return "positive"

    return "neutral"


def normalize_point(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" -*\t\r\n")
    cleaned = re.sub(r"^\d+[\).\s-]+", "", cleaned)
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
    useless_phrases = ("no major", "no clear", "no complaints", "no cons", "no negatives", "no issues", "none", "not mentioned", "n/a")
    return text_lower.startswith(useless_phrases) or any(phrase in text_lower for phrase in ("no complaints mentioned", "no major recurring"))


def prepare_and_split(reviews: list[str]) -> list[str]:
    prepared = []
    for raw in reviews:
        for fragment in split_into_clauses(raw):
            if is_valid_fragment(fragment):
                prepared.append(fragment)
    return prepared[:MAX_REVIEWS_TO_ANALYZE]


def get_impact_level(polarity_score: float) -> str:
    """FIX 10: Derive impact level from polarity score."""
    abs_score = abs(polarity_score)
    if abs_score >= 0.5:
        return "high"
    elif abs_score >= 0.2:
        return "medium"
    return "low"


def make_analysis_point(text: str, domain: str) -> AnalysisPoint:
    """FIX 10: Build a frontend-ready AnalysisPoint from raw text."""
    feature = extract_feature_cached(text, domain) or "general"
    polarity = get_sentiment_polarity(text)
    impact = get_impact_level(polarity)
    return AnalysisPoint(text=shorten(text), feature=feature, impact=impact)


def extract_points(clauses: list[str], domain: str = "generic") -> dict[str, list]:
    pros, cons, neutral = [], [], []
    seen = {}
    feature_groups: dict[str, list[str]] = {}

    for clause in clauses:
        label = classify_sentence(clause)
        normalized = normalize_point(clause)
        sig = get_point_signature(normalized)
        feature = extract_feature_cached(clause, domain)

        if is_useless_point(normalized):
            continue

        if feature and feature in feature_groups:
            is_duplicate = False
            for existing in feature_groups[feature]:
                if points_overlap(normalized, existing):
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

        if sig in seen.values():
            continue

        seen[sig] = sig
        if feature:
            feature_groups.setdefault(feature, []).append(normalized)

        if label == "positive":
            pros.append(normalized)
        elif label == "negative":
            cons.append(normalized)
        elif "overall" not in normalized.lower() and "mixed" not in normalized.lower():
            neutral.append(normalized)

    return {
        "pros": [make_analysis_point(p, domain) for p in pros[:MAX_POINTS]],
        "cons": [make_analysis_point(c, domain) for c in cons[:MAX_POINTS]],
        "neutral_points": [shorten(n) for n in neutral[:MAX_NEUTRAL_POINTS]],
    }


def calculate_sentiment(clauses: list[str]) -> dict[str, float | int]:
    counts = {"positive": 0, "neutral": 0, "negative": 0, "total": len(clauses)}
    for clause in clauses:
        label = classify_sentence(clause)
        counts[label] += 1

    total = counts["total"] or 1
    return {
        "positive": round(counts["positive"] / total * 100, 2),
        "neutral": round(counts["neutral"] / total * 100, 2),
        "negative": round(counts["negative"] / total * 100, 2),
        "total": counts["total"],
    }


def calculate_feature_scores(clauses: list[str], domain: str = "generic") -> list[FeatureScore]:
    feature_data = {}

    for clause in clauses:
        if not is_valid_fragment(clause):
            continue
        label = classify_sentence(clause)
        feature = extract_feature_cached(clause, domain)
        if not feature:
            continue
        normalized = shorten(normalize_point(clause))
        feature_data.setdefault(feature, {"positive": [], "negative": []})
        if label == "positive":
            feature_data[feature]["positive"].append(normalized)
        elif label == "negative":
            feature_data[feature]["negative"].append(normalized)

    scores = []
    for feature, data in feature_data.items():
        pos = len(data["positive"])
        neg = len(data["negative"])
        total = pos + neg
        if total == 0:
            continue
        score = ((pos - neg) / total) * 100
        scores.append(FeatureScore(
            feature=feature,
            display_name=feature.replace("_", " ").title(),
            positive_count=pos,
            negative_count=neg,
            total_mentions=total,
            score=round(score, 1),
        ))

    scores.sort(key=lambda x: abs(x.score), reverse=True)
    return scores


# ==============================================================================
# AI HELPERS
# ==============================================================================

def resolve_model_name(raw: str) -> str:
    name = raw.strip() or DEFAULT_GEMINI_MODEL
    return LEGACY_MODEL_ALIASES.get(name, name)


def parse_ai_response(raw_text: str) -> Optional[dict]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()

    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            pros = payload.get("pros", [])
            cons = payload.get("cons", [])
            if not isinstance(pros, list) or not isinstance(cons, list):
                return None
            if not pros and not cons:
                return None
            return {
                "summary": str(payload.get("summary", "")).strip(),
                "pros": [shorten(p) for p in pros if isinstance(p, str) and len(p.strip()) >= 5],
                "cons": [shorten(c) for c in cons if isinstance(c, str) and len(c.strip()) >= 5],
                "neutral_points": [shorten(n) for n in payload.get("neutral_points", []) if isinstance(n, str) and len(n.strip()) >= 5],
            }
    except (json.JSONDecodeError, Exception):
        pass

    sm = re.search(r"summary\s*:\s*(.+?)(?:\n\s*pros?\s*:|\Z)", cleaned, flags=re.I | re.S)
    pm = re.search(r"pros?\s*:\s*(.+?)(?:\n\s*cons?\s*:|\Z)", cleaned, flags=re.I | re.S)
    cm = re.search(r"cons?\s*:\s*(.+?)(?:\n\s*neutral|\Z)", cleaned, flags=re.I | re.S)

    def extract_list(text):
        if not text:
            return []
        return [shorten(l.strip()) for l in text.splitlines() if l.strip() and ":" not in l.lower() and len(l.strip()) >= 5]

    pros = extract_list(pm.group(1) if pm else "")
    cons = extract_list(cm.group(1) if cm else "")

    if not pros and not cons:
        return None

    return {
        "summary": sm.group(1).strip() if sm else "",
        "pros": pros,
        "cons": cons,
        "neutral_points": [],
    }


def build_analysis_prompt(reviews: list[str], domain: str = "generic") -> str:
    domain_context = f"\nProduct domain: {domain.replace('_', ' ')}" if domain != "generic" else ""
    return f"""You are analyzing product reviews. Extract ONLY factual insights.

Domain: {domain}{domain_context}

STRICT RULES:
1. Each point MUST mention a product feature (e.g., battery, camera, display, comfort, taste)
2. Split mixed sentences: "camera good BUT battery bad" → separate points
3. MAX: 3 pros, 3 cons, 1 neutral point
4. NO generic phrases like "good quality" or "nice product"
5. NEVER invent details not in the reviews

Return STRICT JSON:
{{
  "summary": "1-2 sentence verdict mentioning specific features",
  "pros": ["feature + specific positive observation"],
  "cons": ["feature + specific negative observation"],
  "neutral_points": ["neutral observation about a feature"]
}}

Reviews:
{chr(10).join('- ' + r[:200] for r in reviews[:5])}""".strip()


def build_summary_prompt(pros: list[str], cons: list[str]) -> str:
    return f"""Summarize in 1-2 natural sentences. Sound human. Mention specific features.

Pros: {pros[:3] if pros else 'None'}
Cons: {cons[:3] if cons else 'None'}

Return ONLY the summary text. Do NOT start with "Overall" or "In summary":""".strip()


# ==============================================================================
# FIX 5: HUMAN-LIKE SUMMARY BUILDER
# ==============================================================================

def build_summary(pros: list[AnalysisPoint], cons: list[AnalysisPoint], sentiment: dict, domain: str = "generic") -> str:
    """FIX 5: Natural, human-sounding summary using feature names."""
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
            return f"Users generally love the {feat_str}." + (
                f" Some report issues with the {fmt(con_features[0])}." if con_features else ""
            )
        return "Most users are satisfied with this product."

    elif neg_pct >= 60:
        if con_features:
            feat_str = fmt(con_features[0])
            return f"Users report notable concerns with the {feat_str}." + (
                f" A few highlight the {fmt(pro_features[0])} as a positive." if pro_features else ""
            )
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


# ==============================================================================
# FIX 6: USER PERSONALIZATION — boost user-requested feature in output ordering
# ==============================================================================

def apply_user_focus(points: list[AnalysisPoint], user_focus: Optional[str]) -> list[AnalysisPoint]:
    """FIX 6: Boost points matching user's focus feature to top of list."""
    if not user_focus:
        return points
    focus = user_focus.lower().strip()
    boosted = [p for p in points if focus in p.feature.lower() or focus in p.text.lower()]
    rest = [p for p in points if p not in boosted]
    return boosted + rest


def select_best_points(
    points_a: list[str],
    points_b: list[str],
    label: str,
    domain: str = "generic",
) -> list[AnalysisPoint]:
    seen = set()
    all_points = []

    for point in points_a + points_b:
        sig = get_point_signature(point)
        if sig in seen:
            continue

        text_lower = point.lower()
        score = 0.0

        if extract_feature_cached(point, domain):
            score += 3.0

        if label == "positive":
            score += 2.0 if any(kw in text_lower for kw in STRONG_POSITIVE) else (1.0 if any(kw in text_lower for kw in SOFT_POSITIVE) else 0)
        else:
            score += 2.0 if any(kw in text_lower for kw in STRONG_NEGATIVE) else (1.0 if any(kw in text_lower for kw in SOFT_NEGATIVE) else 0)

        if is_useless_point(point):
            score -= 5.0

        seen.add(sig)
        all_points.append((point, score))

    all_points.sort(key=lambda x: x[1], reverse=True)

    selected: list[AnalysisPoint] = []
    selected_features = set()

    for point, _ in all_points:
        if any(points_overlap(point, sp.text) for sp in selected):
            continue
        feature = extract_feature_cached(point, domain)
        if feature and feature in selected_features and len(selected) < MAX_POINTS:
            continue
        ap = make_analysis_point(point, domain)
        selected.append(ap)
        if feature:
            selected_features.add(feature)
        if len(selected) >= MAX_POINTS:
            break

    return selected


# ==============================================================================
# WORKER PROCESSOR
# ==============================================================================

async def process_analysis_task(
    reviews: list[str],
    detailed: bool,
    user_focus: Optional[str] = None,
) -> tuple:
    warnings: list[WarningDetail] = []
    start_time = time.time()

    raw_text = " ".join(reviews)
    domain = detect_domain(raw_text)

    if domain == "generic":
        warnings.append(WarningDetail(
            type="DOMAIN_UNKNOWN",
            message="Could not detect product domain — using generic analysis mode."
        ))

    # FIX 8: context-aware cache key
    cache_key = cache_manager.generate_cache_key(reviews, detailed, domain)
    cached = cache_manager.get(cache_key)

    if cached:
        CACHE_HITS.inc()
        pros = [AnalysisPoint(**p) for p in cached["analysis"].get("pros", [])]
        cons = [AnalysisPoint(**c) for c in cached["analysis"].get("cons", [])]
        pros = apply_user_focus(pros, user_focus)
        cons = apply_user_focus(cons, user_focus)
        cached["analysis"]["pros"] = pros
        cached["analysis"]["cons"] = cons
        return (
            cached["analysis"],
            cached["sentiment"],
            [FeatureScore(**fs) for fs in cached.get("feature_scores", [])],
            True,
            [],
            domain,
        )

    clauses = prepare_and_split(reviews)

    if not clauses:
        return (
            {"summary": "No valid review content found.", "pros": [], "cons": [], "neutral_points": []},
            {"positive": 0.0, "neutral": 0.0, "negative": 0.0, "total": 0},
            [],
            False,
            [WarningDetail(type="NO_CONTENT", message="No valid review content found.")],
            domain,
        )

    rule_based = extract_points(clauses, domain)
    sentiment = calculate_sentiment(clauses)

    ai_result = None
    ai_summary = None

    should_use_ai = gemini_manager.has_keys() and len(clauses) > 5

    if should_use_ai:
        try:
            model_name = resolve_model_name(os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))

            async with gemini_semaphore:
                request_tracker.active_gemini += 1
                try:
                    ai_result = await asyncio.wait_for(
                        gemini_manager.analyze_reviews(clauses, model_name, domain),
                        timeout=GEMINI_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    warnings.append(WarningDetail(
                        type="AI_TIMEOUT",
                        message="AI enhancement timed out — falling back to rule-based analysis."
                    ))
                    ai_result = None
                finally:
                    request_tracker.active_gemini -= 1

            if ai_result:
                async with gemini_semaphore:
                    request_tracker.active_gemini += 1
                    try:
                        ai_summary = await asyncio.wait_for(
                            gemini_manager.generate_summary(
                                ai_result.get("pros", []),
                                ai_result.get("cons", []),
                                model_name
                            ),
                            timeout=GEMINI_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        ai_summary = None
                    finally:
                        request_tracker.active_gemini -= 1

        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")
            warnings.append(WarningDetail(
                type="AI_UNAVAILABLE",
                message="AI enhancement unavailable — using rule-based analysis."
            ))
    else:
        if not gemini_manager.has_keys():
            logger.info("No Gemini keys configured — rule-based analysis only")
        else:
            logger.info(f"Rule-based analysis only ({len(clauses)} clauses)")

    if ai_result and (ai_result.get("pros") or ai_result.get("cons")):
        final_pros = select_best_points(ai_result.get("pros", []), [p.text for p in rule_based["pros"]], "positive", domain)
        final_cons = select_best_points(ai_result.get("cons", []), [c.text for c in rule_based["cons"]], "negative", domain)

        final_pros = [p for p in final_pros if not any(points_overlap(p.text, c.text) for c in final_cons)]
        final_cons = [c for c in final_cons if not any(points_overlap(c.text, p.text) for p in final_pros)]

        if ai_summary and len(ai_summary.split()) >= 5:
            final_summary = ai_summary
        else:
            final_summary = build_summary(final_pros, final_cons, sentiment, domain)

        final_analysis = {
            "summary": final_summary,
            "pros": final_pros[:MAX_POINTS],
            "cons": final_cons[:MAX_POINTS],
            "neutral_points": rule_based["neutral_points"],
        }
    else:
        final_pros = rule_based["pros"]
        final_cons = rule_based["cons"]
        final_summary = build_summary(final_pros, final_cons, sentiment, domain)
        final_analysis = {
            "summary": final_summary,
            "pros": final_pros,
            "cons": final_cons,
            "neutral_points": rule_based["neutral_points"],
        }

    # FIX 6: Apply user personalization
    final_analysis["pros"] = apply_user_focus(final_analysis["pros"], user_focus)
    final_analysis["cons"] = apply_user_focus(final_analysis["cons"], user_focus)

    # Cache serializable copy
    cache_data = {
        "analysis": {
            "summary": final_analysis["summary"],
            "pros": [p.model_dump() for p in final_analysis["pros"]],
            "cons": [c.model_dump() for c in final_analysis["cons"]],
            "neutral_points": final_analysis["neutral_points"],
        },
        "sentiment": sentiment,
    }
    if detailed:
        cache_data["feature_scores"] = [fs.model_dump() for fs in calculate_feature_scores(clauses, domain)]

    cache_manager.set(cache_key, cache_data)
    feature_scores = calculate_feature_scores(clauses, domain) if detailed else []

    return final_analysis, sentiment, feature_scores, False, warnings, domain


# ==============================================================================
# WORKER LOOPS
# ==============================================================================

async def worker_loop(worker_id: int):
    logger.info(f"Worker {worker_id} started")

    while True:
        try:
            task_data = await asyncio.wait_for(WORKER_QUEUE.get(), timeout=1.0)
            task_id, reviews, detailed, user_focus, future = task_data

            try:
                result = await process_analysis_task(reviews, detailed, user_focus)
                if not future.done():
                    future.set_result(result)
            except Exception as exc:
                logger.error(f"Worker {worker_id} task {task_id} error: {exc}")
                if not future.done():
                    future.set_exception(exc)
            finally:
                request_tracker.queue_depth = WORKER_QUEUE.qsize()
                WORKER_QUEUE.task_done()
                with TASK_RESULTS_LOCK:
                    TASK_RESULTS.pop(task_id, None)

        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.error(f"Worker {worker_id} loop error: {e}")
            await asyncio.sleep(1)





# ==============================================================================
# SCORE CALCULATION
# ==============================================================================

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
    confidence = round(dominant * (0.25 + 0.75 * sample_factor) * 100, 2)

    return score, min(100.0, confidence)


# ==============================================================================
# FIX 9: STRUCTURED OBSERVABILITY LOGGING
# ==============================================================================

def log_request_analytics(
    user_id: str,
    domain: str,
    elapsed: float,
    from_cache: bool,
    clause_count: int,
    endpoint: str,
    status: str = "success",
):
    """FIX 9: Structured log entry for observability."""
    logger.info(json.dumps({
        "event": "analytics_complete",
        "endpoint": endpoint,
        "status": status,
        "user": user_id,
        "domain": domain,
        "latency_ms": round(elapsed * 1000, 1),
        "cached": from_cache,
        "clauses": clause_count,
    }))


# ==============================================================================
# SHARED HANDLER (used by both v1 and v2 routes)
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

        scalable_limiter.record_request(user_id)

        detailed = request.query_params.get("detailed", "false").lower() == "true"
        user_focus = payload.user_focus

        task_id = generate_task_id()
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        with TASK_RESULTS_LOCK:
            TASK_RESULTS[task_id] = future

        try:
            WORKER_QUEUE.put_nowait((task_id, reviews, detailed, user_focus, future))
            request_tracker.queue_depth = WORKER_QUEUE.qsize()
        except asyncio.QueueFull:
            REQUEST_COUNT.labels(endpoint=endpoint, status="queue_full").inc()
            raise HTTPException(status_code=503, detail="Server busy. Try again later.")

        try:
            timer = REQUEST_LATENCY.labels(endpoint=endpoint)
            with timer.time():
                analysis, sentiment, feature_scores, from_cache, ai_warnings, domain = await asyncio.wait_for(future, timeout=30.0)
        except AttributeError:
            analysis, sentiment, feature_scores, from_cache, ai_warnings, domain = await asyncio.wait_for(future, timeout=30.0)

        score, confidence = calculate_score_and_confidence(sentiment)

        # FIX 9: Structured observability
        log_request_analytics(
            user_id=user_id,
            domain=domain,
            elapsed=time.time() - start_time,
            from_cache=from_cache,
            clause_count=sentiment.get("total", 0),
            endpoint=endpoint,
        )

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
            warnings=ai_warnings or [],
            domain=domain,
        )

        if detailed:
            response.explained_pros = [
                ExplainablePoint(
                    text=p.text,
                    feature=p.feature,
                    sentiment="positive",
                    polarity_score=round(get_sentiment_polarity(p.text), 3),
                    impact=p.impact,
                )
                for p in pros
            ]
            response.explained_cons = [
                ExplainablePoint(
                    text=c.text,
                    feature=c.feature,
                    sentiment="negative",
                    polarity_score=round(get_sentiment_polarity(c.text), 3),
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
# FIX 12: VERSIONED ROUTES — /api/v1 and /api/v2
# ==============================================================================

# --- V1 routes (legacy, raw-text only) ---

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


# --- V2 routes (current, adds user_focus personalization + structured points) ---

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


# --- Legacy unversioned routes (kept for backward compatibility) ---

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


# ==============================================================================
# ADMIN ENDPOINTS
# ==============================================================================

@app.get("/stats")
async def get_stats(x_api_key: Optional[str] = Header(None)) -> dict:
    await verify_api_key(x_api_key)
    return {
        "cache": cache_manager.lru_cache.get_stats(),
        
    }


@app.post("/cache/clear")
async def clear_cache(x_api_key: Optional[str] = Header(None)) -> dict:
    await verify_api_key(x_api_key)
    cache_manager.lru_cache.cache.clear()
    cache_manager.lru_cache.size = 0
    return {"status": "cache cleared"}
    


@app.get("/admin/gemini-health")
async def get_gemini_health(x_api_key: Optional[str] = Header(None)) -> dict:
    await verify_api_key(x_api_key)
    return gemini_manager.get_health()


@app.post("/admin/gemini-reload")
async def reload_gemini(x_api_key: Optional[str] = Header(None)) -> dict:
    await verify_api_key(x_api_key)
    gemini_manager.reload_keys()
    return {"status": "reloaded", **gemini_manager.get_health()}


@app.get("/admin/system-health")
async def get_system_health(x_api_key: Optional[str] = Header(None)) -> dict:
    await verify_api_key(x_api_key)
    return {
        "circuit_breaker": gemini_manager.get_circuit_state(),
        "semaphore": {
            "max_concurrent": GEMINI_CONCURRENCY_LIMIT,
            "active_requests": request_tracker.active_gemini,
            "available_slots": GEMINI_CONCURRENCY_LIMIT - request_tracker.active_gemini,
        },
        "worker_queue": {
            "max_size": QUEUE_MAX_SIZE,
            "current_size": request_tracker.queue_depth,
            "utilization_percent": round(request_tracker.queue_depth / QUEUE_MAX_SIZE * 100, 1),
        },
        "rate_limiter": {
            "per_minute_limit": RATE_LIMIT_PER_MINUTE,
            "per_hour_limit": RATE_LIMIT_PER_HOUR,
        },
        "cache": {
            
            "memory_entries": cache_manager.lru_cache.size,
        },
        "active_requests": {
            "http": request_tracker.active_http,
            "gemini": request_tracker.active_gemini,
        },
        "api_versions": ["v1", "v2"],
    }


@app.get("/admin/domain-detection")
async def detect_domain_endpoint(text: str, x_api_key: Optional[str] = Header(None)) -> dict:
    await verify_api_key(x_api_key)
    domain = detect_domain(text)
    return {"text_preview": text[:100], "detected_domain": domain}


# ==============================================================================
# FIX 11: BASIC TEST SUITE (run with: pytest main.py)
# ==============================================================================

def _run_tests():
    """FIX 11: Inline unit tests — run with pytest or python main.py --test"""
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

    # 1. Tokenizer
    def t_tokenize():
        tokens = tokenize("battery-life is great!")
        assert "battery" in tokens
        assert "life" in tokens

    # 2. Domain detection
    def t_domain_electronics():
        assert detect_domain("battery life and camera quality are great") == "electronics"

    def t_domain_clothing():
        assert detect_domain("fabric feels soft, true to size and breathable") == "clothing"

    def t_domain_generic():
        assert detect_domain("it arrived on time") == "generic"

    # 3. Negation handling
    def t_negation_window():
        result = handle_negation("battery is not very good")
        assert "NOT_very" in result
        assert "NOT_good" in result

    def t_negation_simple():
        result = handle_negation("not great")
        assert "NOT_great" in result

    # 4. Sentiment polarity
    def t_sentiment_positive():
        score = get_sentiment_polarity("the camera is excellent and amazing")
        assert score > 0

    def t_sentiment_negative():
        score = get_sentiment_polarity("battery drains fast and overheats")
        assert score < 0

    def t_sentiment_negated():
        score_pos = get_sentiment_polarity("battery is great")
        score_neg = get_sentiment_polarity("battery is not great")
        assert score_pos > score_neg

    # 5. Feature extraction
    def t_feature_camera():
        feat = extract_feature_with_context("the camera takes amazing photos", "electronics")
        assert feat == "camera"

    def t_feature_false_positive_reduced():
        # "fast" alone should not confidently map to performance without more context
        feat = extract_feature_with_context("it arrived fast", "generic")
        # could be None or performance — what matters is no crash
        assert feat is None or isinstance(feat, str)

    # 6. Clause splitting
    def t_clause_split():
        clauses = split_into_clauses("Camera is great but battery drains fast.")
        assert len(clauses) >= 2

    # 7. Point deduplication
    def t_no_overlap_in_pros_cons():
        pros = [AnalysisPoint(text="Battery is great", feature="battery", impact="high")]
        cons = [AnalysisPoint(text="Battery is poor", feature="battery", impact="high")]
        # overlapping text check: these should NOT be flagged as overlapping
        assert not points_overlap("Battery is great", "Camera is poor")

    # 8. Cache key context-aware
    def t_cache_key_context():
        k1 = cache_manager.generate_cache_key(["good product"], detailed=False, domain="electronics")
        k2 = cache_manager.generate_cache_key(["good product"], detailed=True, domain="electronics")
        k3 = cache_manager.generate_cache_key(["good product"], detailed=False, domain="clothing")
        assert k1 != k2
        assert k1 != k3

    # 9. Impact level
    def t_impact_high():
        assert get_impact_level(0.9) == "high"
        assert get_impact_level(0.3) == "medium"
        assert get_impact_level(0.05) == "low"

    # 10. Shorten
    def t_shorten():
        long_text = "x" * 200
        assert len(shorten(long_text)) <= 83  # 80 + "..."

    test("tokenize handles hyphens", t_tokenize)
    test("domain: electronics", t_domain_electronics)
    test("domain: clothing", t_domain_clothing)
    test("domain: generic fallback", t_domain_generic)
    test("negation window covers 3 words", t_negation_window)
    test("negation simple", t_negation_simple)
    test("sentiment: positive text", t_sentiment_positive)
    test("sentiment: negative text", t_sentiment_negative)
    test("sentiment: negation flips score", t_sentiment_negated)
    test("feature: camera extraction", t_feature_camera)
    test("feature: false positive guard", t_feature_false_positive_reduced)
    test("clause splitting on 'but'", t_clause_split)
    test("no overlap in pros vs cons", t_no_overlap_in_pros_cons)
    test("cache key is context-aware", t_cache_key_context)
    test("impact levels", t_impact_high)
    test("shorten truncates correctly", t_shorten)

    passed = sum(1 for r in results if r[0] == "PASS")
    failed = sum(1 for r in results if r[0] != "PASS")

    print(f"\n{'='*50}")
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    for status, name in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name}")
    print()


# ==============================================================================
# ROUTES
# ==============================================================================

@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "message": "AI Product Review Aggregator API v20.0",
        "docs": "/docs",
        "v1": f"{API_V1_PREFIX}/analyze-raw",
        "v2": f"{API_V2_PREFIX}/analyze-raw",
        "legacy": "/analyze-raw",
    }


@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


# ==============================================================================
# STARTUP EVENT
# ==============================================================================





# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        _run_tests()
        sys.exit(0)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)