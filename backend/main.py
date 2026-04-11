import asyncio
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
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

# ==============================================================================
# OPTIONAL VADER SENTIMENT (Fast alternative to TextBlob)
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
# RATE LIMITER
# ==============================================================================

limiter = Limiter(key_func=get_remote_address)

# ==============================================================================
# CONFIGURATION (Dynamic from .env)
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

# Cache settings
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1000"))

# Redis settings
REDIS_URL = os.getenv("REDIS_URL", "")
USE_REDIS = bool(REDIS_URL)

redis_client = None
if USE_REDIS:
    try:
        import redis
        redis_client = redis.from_url(REDIS_URL)
        logger.info("Redis connected")
    except ImportError:
        logger.warning("Redis not installed, using memory fallback")
        redis_client = None

# Constants (✅ Now configurable via environment)
PLACEHOLDER_API_KEYS = {"your_gemini_api_key_here", "replace_with_real_key"}
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LEGACY_MODEL_ALIASES = {"gemini-pro": DEFAULT_GEMINI_MODEL}

# ✅ Dynamic limits from .env
MAX_POINTS = int(os.getenv("MAX_POINTS", "6"))
MAX_POINTS_PER_REVIEW = int(os.getenv("MAX_POINTS_PER_REVIEW", "2"))
MAX_NEUTRAL_POINTS_PER_REVIEW = int(os.getenv("MAX_NEUTRAL_POINTS_PER_REVIEW", "2"))
MAX_REVIEWS = int(os.getenv("MAX_REVIEWS", "100"))
MAX_REVIEWS_TO_ANALYZE = int(os.getenv("MAX_REVIEWS_TO_ANALYZE", "30"))
MAX_REVIEWS_PER_REQUEST = int(os.getenv("MAX_REVIEWS_PER_REQUEST", "20"))
MIN_REVIEW_CHARS = int(os.getenv("MIN_REVIEW_CHARS", "15"))
MIN_REVIEW_WORDS = int(os.getenv("MIN_REVIEW_WORDS", "3"))
MIN_REVIEW_LENGTH = int(os.getenv("MIN_REVIEW_LENGTH", "10"))
MIN_POINT_CHARS = int(os.getenv("MIN_POINT_CHARS", "8"))
SENTIMENT_POLARITY_THRESHOLD = float(os.getenv("SENTIMENT_POLARITY_THRESHOLD", "0.1"))

# Abuse Protection (✅ Configurable)
SPAM_SIMILARITY_THRESHOLD = float(os.getenv("SPAM_SIMILARITY_THRESHOLD", "0.95"))
SPAM_SAME_REVIEW_LIMIT = int(os.getenv("SPAM_SAME_REVIEW_LIMIT", "3"))

# Cooldown
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "300"))

# Rate Limiting (✅ Configurable)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "100"))


# ==============================================================================
# KEYWORD SETS
# ==============================================================================

STRONG_NEGATIVE_KEYWORDS = frozenset({
    "bad", "poor", "worst", "waste", "overheat", "overheating",
    "lag", "lagging", "drain", "drains", "heats", "heating",
    "fails", "failure", "crash", "buggy", "terrible", "horrible",
    "awful", "broken",
})

STRONG_POSITIVE_KEYWORDS = frozenset({
    "excellent", "great", "smooth", "easy", "premium", "bright",
    "sharp", "clean", "love", "best", "amazing",
})

SOFT_NEGATIVE_KEYWORDS = frozenset({
    "slow", "weak", "issue", "problem", "expensive", "overpriced",
    "noisy", "hot", "heats", "drains", "drain", "not enough",
    "not good", "not clear", "not loud", "not impressive",
})

SOFT_POSITIVE_KEYWORDS = frozenset({
    "good", "nice", "fast", "clear", "lightweight", "sleek",
    "fine", "okay", "decent", "works",
})

ALL_POSITIVE_CUES = STRONG_POSITIVE_KEYWORDS | SOFT_POSITIVE_KEYWORDS
ALL_NEGATIVE_CUES = STRONG_NEGATIVE_KEYWORDS | SOFT_NEGATIVE_KEYWORDS

FEATURE_WEIGHTS = {
    "battery": 3.0, "performance": 3.0, "camera": 3.0,
    "display": 2.0, "charging": 2.0, "software": 2.0,
    "sound": 2.0, "build": 2.0, "price": 2.0, "support": 2.0,
    "design": 1.0, "comfort": 1.0, "portability": 1.0, "connectivity": 1.0,
}

FEATURE_ANCHORS = frozenset({
    "battery", "camera", "performance", "design", "display",
    "sound", "charging", "build", "price", "software", "support",
    "comfort", "brightness", "volume", "speaker", "screen",
    "quality", "audio", "setup", "processor", "ram", "storage",
    "gaming", "touch", "network", "signal", "update", "wifi",
    "bluetooth", "port", "charger", "cable", "case", "size",
    "weight", "ergonomics", "haptic", "vibration",
})

FEATURE_ALIAS_MAP = {
    "battery": {"battery", "backup", "drain", "drains", "mah", "power"},
    "camera": {"camera", "photo", "photos", "video", "lens", "focus", "zoom", "portrait", "selfie"},
    "performance": {"performance", "lag", "lagging", "slow", "speed", "processor", "ram", "gaming", "fps"},
    "design": {"design", "look", "looks", "style", "lightweight", "sleek", "color", "colors"},
    "display": {"display", "screen", "brightness", "touch", "refresh", "oled", "lcd", "resolution"},
    "sound": {"sound", "audio", "speaker", "mic", "microphone", "volume", "noise", "bass", "treble"},
    "charging": {"charging", "charge", "charger", "charging speed", "fast charge"},
    "build": {"build", "quality", "durable", "durability", "material", "plastic", "metal", "glass"},
    "price": {"price", "cost", "expensive", "overpriced", "value", "worth"},
    "software": {"software", "ui", "ux", "update", "app", "apps", "os", "android", "ios"},
    "support": {"support", "service", "warranty", "customer"},
    "comfort": {"comfort", "comfortable", "fit", "pain", "ear", "earbuds", "headphones"},
    "connectivity": {"wifi", "bluetooth", "signal", "network", "connection"},
    "portability": {"size", "weight", "lightweight", "portable", "bulk", "heavy"},
}

COMPARISON_STOP_WORDS = frozenset({
    "a", "an", "and", "as", "at", "for", "from",
    "in", "is", "of", "the", "to", "very", "with",
})

GENERIC_POINT_MARKERS = (
    "no major", "no clear", "no complaints", "no complaint",
    "no cons", "no negatives", "no issues", "none",
    "not mentioned", "n/a",
)

USELESS_POINT_PHRASES = (
    "no major recurring strengths", "no major recurring complaints",
    "no complaints mentioned", "no complaint mentioned",
    "no complaints", "no issues",
)

PRODUCT_KEYWORDS = (
    "battery", "camera", "performance", "design", "quality", "price",
    "screen", "display", "speaker", "software", "support", "charging",
    "build", "setup", "brightness", "volume", "sound", "audio",
)

SENTIMENT_WORDS = (
    "good", "bad", "poor", "excellent", "slow", "fast", "great",
    "amazing", "terrible", "buggy", "average",
)

QUESTION_PREFIXES = (
    "who", "what", "when", "where", "why", "how",
    "tell me", "explain", "define",
)


# ==============================================================================
# PYDANTIC MODELS
# ==============================================================================

class ReviewRequest(BaseModel):
    reviews: list[str] = Field(..., min_length=1)

    @field_validator("reviews")
    @classmethod
    def clean_reviews(cls, reviews: list[str]) -> list[str]:
        cleaned = [r.strip() for r in reviews if r and r.strip()]
        if not cleaned:
            raise ValueError("Please provide at least one non-empty review.")
        if len(cleaned) > MAX_REVIEWS:
            raise ValueError(f"Please provide no more than {MAX_REVIEWS} reviews at a time.")
        return cleaned


class SentimentBreakdown(BaseModel):
    positive: float
    neutral: float
    negative: float
    total: int


class ExplainablePoint(BaseModel):
    text: str
    feature: str
    sentiment: str
    trigger_words: list[str]
    polarity_score: float
    confidence: float


class FeatureScore(BaseModel):
    feature: str
    display_name: str
    positive_count: int
    negative_count: int
    neutral_count: int
    total_mentions: int
    score: float
    weight: float
    weighted_score: float
    examples_positive: list[str]
    examples_negative: list[str]


class AIAnalysis(BaseModel):
    summary: str
    pros: list[str]
    cons: list[str]
    neutral_points: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    summary: str
    pros: list[str]
    cons: list[str]
    neutral_points: list[str] = Field(default_factory=list)
    sentiment: SentimentBreakdown
    score: float
    confidence: float
    explained_pros: list[ExplainablePoint] = Field(default_factory=list)
    explained_cons: list[ExplainablePoint] = Field(default_factory=list)
    feature_scores: list[FeatureScore] = Field(default_factory=list)
    cached: bool = Field(default=False)


# ==============================================================================
# FASTAPI APP SETUP
# ==============================================================================

app = FastAPI(
    title="AI Product Review Aggregator API",
    description="Production-ready review analysis with full security features",
    version="9.4",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

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
        logger.warning("No API keys configured - running in open mode!")
        return "dev"

    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required. Pass 'X-API-Key' header.")

    if x_api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API key attempt: {x_api_key[:8]}...")
        raise HTTPException(status_code=401, detail="Invalid API key.")

    return x_api_key


# ==============================================================================
# LRU CACHE WITH MAX SIZE
# ==============================================================================

class LRUCache:
    """LRU Cache with max size limit and eviction."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0

    def _evict_if_needed(self):
        while len(self.cache) >= self.max_size:
            if self.cache:
                self.cache.popitem(last=False)
                self.eviction_count += 1

    def get(self, key: str) -> Optional[dict]:
        if key not in self.cache:
            self.miss_count += 1
            return None

        data_str, expiry = self.cache[key]

        if time.time() > expiry:
            del self.cache[key]
            self.miss_count += 1
            return None

        self.cache.move_to_end(key)
        self.hit_count += 1
        return json.loads(data_str)

    def set(self, key: str, data: dict, ttl: int = 3600):
        self._evict_if_needed()

        data_str = json.dumps(data)
        expiry = time.time() + ttl

        if key in self.cache:
            del self.cache[key]

        self.cache[key] = (data_str, expiry)
        self.cache.move_to_end(key)

    def get_stats(self) -> dict:
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        return {
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": f"{hit_rate:.1f}%",
            "entries": len(self.cache),
            "max_size": self.max_size,
            "evictions": self.eviction_count,
        }


class CacheManager:
    """Cache manager with Redis + LRU memory fallback."""

    def __init__(self):
        self.lru_cache = LRUCache(max_size=MAX_CACHE_SIZE)

    def generate_cache_key(self, reviews: list[str], api_key: str) -> str:
        reviews_hash = hashlib.sha256(json.dumps(reviews, sort_keys=True).encode()).hexdigest()
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        return f"review_cache:{key_hash}:{reviews_hash[:32]}"

    def get(self, key: str) -> Optional[dict]:
        if not ENABLE_CACHE:
            return None

        if redis_client:
            try:
                data = redis_client.get(key)
                if data:
                    self.lru_cache.hit_count += 1
                    return json.loads(data)
            except Exception as e:
                logger.warning("Redis get failed: %s", e)

        return self.lru_cache.get(key)

    def set(self, key: str, data: dict, ttl: int = None):
        if not ENABLE_CACHE:
            return

        ttl = ttl or CACHE_TTL_SECONDS
        data_str = json.dumps(data)

        if redis_client:
            try:
                redis_client.setex(key, ttl, data_str)
                return
            except Exception as e:
                logger.warning("Redis set failed: %s", e)

        self.lru_cache.set(key, data, ttl)

    def get_stats(self) -> dict:
        return {
            "lru_cache": self.lru_cache.get_stats(),
            "redis_connected": redis_client is not None,
        }
    
    def clear_redis_cache(self):
        if not redis_client:
            return 0
        
        deleted = 0
        cursor = 0
        pattern = "review_cache:*"
        
        try:
            while True:
                cursor, keys = redis_client.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    redis_client.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning("Redis cache clear failed: %s", e)
        
        return deleted


cache_manager = CacheManager()


# ==============================================================================
# COOLDOWN PROTECTION
# ==============================================================================

class CooldownProtection:
    """Cooldown protection with Redis + memory fallback."""

    def __init__(self):
        self.cooldown_seconds = COOLDOWN_SECONDS
        self.violation_count: dict[str, list[float]] = {}
        self.blocked_until: dict[str, float] = {}

    def _get_identifier(self, api_key: str, ip: str) -> str:
        return f"{api_key}:{ip}"

    def is_blocked(self, api_key: str, ip: str) -> tuple[bool, int]:
        identifier = self._get_identifier(api_key, ip)
        current_time = time.time()

        if redis_client:
            try:
                blocked_key = f"blocked:{identifier}"
                ttl = redis_client.ttl(blocked_key)
                if ttl > 0:
                    return True, ttl
                redis_client.delete(blocked_key)
            except Exception as e:
                logger.warning("Redis cooldown check failed: %s", e)

        if identifier in self.blocked_until:
            unblock_time = self.blocked_until[identifier]
            if current_time < unblock_time:
                return True, int(unblock_time - current_time)
            else:
                del self.blocked_until[identifier]

        return False, 0

    def record_abuse(self, api_key: str, ip: str, reason: str):
        identifier = self._get_identifier(api_key, ip)
        current_time = time.time()

        if identifier in self.violation_count:
            self.violation_count[identifier] = [
                t for t in self.violation_count[identifier]
                if current_time - t < 3600
            ]

        if identifier not in self.violation_count:
            self.violation_count[identifier] = []
        self.violation_count[identifier].append(current_time)

        violation_count = len(self.violation_count[identifier])

        block_duration = 300
        if violation_count == 1:
            block_duration = 60
        elif violation_count == 2:
            block_duration = 300
        elif violation_count == 3:
            block_duration = 900
        else:
            block_duration = 1800

        unblock_time = current_time + block_duration

        if redis_client:
            try:
                blocked_key = f"blocked:{identifier}"
                redis_client.setex(blocked_key, block_duration, "1")
                violation_key = f"violations:{identifier}"
                redis_client.setex(violation_key, 3600, str(violation_count))
            except Exception as e:
                logger.warning("Redis block record failed: %s", e)

        self.blocked_until[identifier] = unblock_time

        logger.warning(
            f"ABUSE - {identifier}: {reason}. Violation #{violation_count}. "
            f"Blocked for {block_duration}s"
        )


cooldown_protection = CooldownProtection()


# ==============================================================================
# RATE LIMITING
# ==============================================================================

class ScalableRateLimiter:
    """Redis-backed rate limiter with memory fallback."""

    def __init__(self):
        self.cleanup_interval = 3600
        self.last_cleanup = time.time()
        self._requests = {}

    def _cleanup_old_data(self):
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return

        if redis_client:
            try:
                cursor = 0
                while True:
                    cursor, keys = redis_client.scan(cursor=cursor, match="rate:requests:*", count=100)
                    for key in keys:
                        if redis_client.ttl(key) <= 0:
                            redis_client.delete(key)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning("Redis cleanup failed: %s", e)

        self.last_cleanup = current_time

    def record_request(self, identifier: str):
        self._cleanup_old_data()

        if redis_client:
            try:
                key = f"rate:requests:{identifier}"
                now = time.time()
                pipe = redis_client.pipeline()
                pipe.lpush(key, now)
                pipe.ltrim(key, 0, 999)
                pipe.expire(key, 3600)
                pipe.execute()
                return
            except Exception as e:
                logger.warning("Redis record failed: %s", e)

        if identifier not in self._requests:
            self._requests[identifier] = []
        self._requests[identifier].append(time.time())

    def get_request_count(self, identifier: str, window_seconds: int = 60) -> int:
        self._cleanup_old_data()

        if redis_client:
            try:
                key = f"rate:requests:{identifier}"
                cutoff = time.time() - window_seconds
                redis_client.ltrim(key, 0, 999)
                requests = redis_client.lrange(key, 0, -1)
                return sum(1 for r in requests if float(r) > cutoff)
            except Exception as e:
                logger.warning("Redis count failed: %s", e)

        if identifier not in self._requests:
            return 0

        cutoff = time.time() - window_seconds
        self._requests[identifier] = [t for t in self._requests[identifier] if t > cutoff]
        return len(self._requests[identifier])

    def is_rate_limited(self, identifier: str, per_minute: int = None, per_hour: int = None) -> tuple[bool, str]:
        per_minute = per_minute or RATE_LIMIT_PER_MINUTE
        per_hour = per_hour or RATE_LIMIT_PER_HOUR
        
        minute_count = self.get_request_count(identifier, 60)
        hour_count = self.get_request_count(identifier, 3600)

        if minute_count >= per_minute:
            return True, f"Per-minute limit exceeded ({minute_count}/{per_minute})"

        if hour_count >= per_hour:
            return True, f"Per-hour limit exceeded ({hour_count}/{per_hour})"

        return False, ""


scalable_limiter = ScalableRateLimiter()


# ==============================================================================
# GEMINI CLIENT MANAGER v3.0
# ==============================================================================

class GeminiKeyConfig:
    """Thread-safe configuration for a single Gemini API key."""

    def __init__(self, key: str, name: str = ""):
        self._key = key.strip()
        self._hash = self._compute_hash()
        self._name = name or f"key_{self._hash[:6]}"

        self._is_available = True
        self._failure_count = 0
        self._last_failure: Optional[float] = None
        self._success_count = 0
        self._total_requests = 0

        self._failure_threshold = 5
        self._recovery_timeout = 300

    def _compute_hash(self) -> str:
        return hashlib.sha256(self._key.encode()).hexdigest()

    @property
    def key(self) -> str:
        return self._key

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> str:
        return self._hash[:6]

    @property
    def hash(self) -> str:
        return self._hash

    @property
    def is_healthy(self) -> bool:
        if not self._is_available:
            if self._last_failure and (time.time() - self._last_failure) > self._recovery_timeout:
                self._is_available = True
                self._failure_count = 0
                logger.info(f"Key {self._name} auto-recovered")
                return True
            return False
        return True

    @property
    def is_rate_limited(self) -> bool:
        if self._last_failure and self._failure_count >= 3:
            if (time.time() - self._last_failure) < 60:
                return True
        return False

    @property
    def stats(self) -> dict:
        total = self._total_requests or 1
        return {
            "id": self.id,
            "name": self.name,
            "healthy": self.is_healthy,
            "rate_limited": self.is_rate_limited,
            "failures": self._failure_count,
            "successes": self._success_count,
            "total_requests": self._total_requests,
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
            logger.warning(
                f"Circuit OPEN: Key {self._name} "
                f"(failures: {self._failure_count}, rate_limit: {is_rate_limit})"
            )


class GeminiClientManager:
    """Production-ready multi-key Gemini client manager."""

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
        self._rotation_lock = threading.Lock()
        self._index = 0
        self._round_robin: list[str] = []

        self._max_retries = 3
        self._base_delay = 1.0
        self._max_delay = 30.0
        self._consecutive_failures = 0
        self._circuit_open_time: Optional[float] = None

        self._load_keys()
        logger.info(f"GeminiClientManager initialized: {len(self._keys)} key(s)")

    def has_keys(self) -> bool:
        return bool(self._keys)

    def get_healthy_key_count(self) -> int:
        return sum(1 for cfg in self._keys.values() if cfg.is_healthy)

    def is_circuit_open(self) -> bool:
        if self._circuit_open_time is None:
            return False
        if time.time() - self._circuit_open_time > 60:
            self._reset_circuit()
            return False
        return True

    def _reset_circuit(self):
        self._circuit_open_time = None
        self._consecutive_failures = 0

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
                    key = key.strip()
                    self._add_key(key, name.strip(), existing_hashes)
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
            logger.debug(f"Skipping duplicate key: {key_hash[:6]}...")
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

    def _get_next_key(self) -> Optional[GeminiKeyConfig]:
        with self._rotation_lock:
            if not self._round_robin or self.is_circuit_open():
                return None

            attempts = 0
            while attempts < len(self._round_robin):
                key_hash = self._round_robin[self._index]
                self._index = (self._index + 1) % len(self._round_robin)

                if key_hash in self._keys and self._keys[key_hash].is_healthy:
                    return self._keys[key_hash]

                attempts += 1

            self._rebuild_rotation()

            if not self._round_robin:
                return None

            key_hash = self._round_robin[self._index % len(self._round_robin)]
            return self._keys.get(key_hash)

    def _calculate_delay(self, attempt: int, is_rate_limit: bool = False) -> float:
        base = self._base_delay * (2 ** attempt)
        if is_rate_limit:
            base *= 3
        jitter = random.uniform(0.75, 1.25)
        return min(base * jitter, self._max_delay)

    def _classify_error(self, error: Exception) -> tuple[bool, bool]:
        msg = str(error).lower()

        rate_limit_indicators = {
            "429", "rate limit", "quota", "too many requests",
            "resource exhausted", "limit exceeded"
        }

        if any(p in msg for p in rate_limit_indicators):
            return True, True

        transient_indicators = {
            "500", "502", "503", "504", "timeout", "timed out",
            "connection", "unavailable", "internal error"
        }

        if any(p in msg for p in transient_indicators):
            return True, False

        return False, False

    async def _execute_with_retry(self, func, *args, **kwargs) -> tuple[Optional[object], Optional[Exception]]:
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            key_config = self._get_next_key()

            if not key_config:
                self._consecutive_failures += 1

                if self._consecutive_failures >= 3 and self._circuit_open_time is None:
                    self._circuit_open_time = time.time()
                    logger.error(
                        f"CIRCUIT OPEN - All keys failing. Pausing 60s. "
                        f"Failures: {self._consecutive_failures}"
                    )

                return None, Exception("No healthy API keys available")

            try:
                client = genai.Client(api_key=key_config.key)
                result = await asyncio.to_thread(func, client, *args, **kwargs)

                key_config.record_success()
                self._consecutive_failures = 0
                return result, None

            except Exception as e:
                last_error = e
                is_retryable, is_rl = self._classify_error(e)

                if is_rl:
                    key_config.record_failure(is_rate_limit=True)
                elif is_retryable:
                    key_config.record_failure(is_rate_limit=False)
                else:
                    return None, e

                logger.warning(f"Key {key_config.id} failed (attempt {attempt + 1}): {str(e)[:80]}")

                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._calculate_delay(attempt, is_rl))
                    self._rebuild_rotation()

        return None, last_error

    async def analyze_reviews(self, reviews: list[str], model_name: str = DEFAULT_GEMINI_MODEL) -> Optional[dict]:
        def _call(client):
            return client.models.generate_content(
                model=model_name,
                contents=build_analysis_prompt(reviews),
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                ),
            )

        result, error = await self._execute_with_retry(_call)

        if error or not result:
            logger.error(f"Gemini analyze failed: {error}")
            return None

        return parse_ai_response(result.text or "")

    async def generate_summary(self, pros: list[str], cons: list[str], model_name: str = DEFAULT_GEMINI_MODEL) -> Optional[str]:
        def _call(client):
            return client.models.generate_content(
                model=model_name,
                contents=build_summary_prompt(pros, cons),
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="text/plain"
                ),
            )

        result, error = await self._execute_with_retry(_call)

        if error or not result:
            logger.error(f"Gemini summary failed: {error}")
            return None

        return (result.text or "").strip()

    def get_health(self) -> dict:
        return {
            "total_keys": len(self._keys),
            "healthy_keys": self.get_healthy_key_count(),
            "circuit_open": self.is_circuit_open(),
            "consecutive_failures": self._consecutive_failures,
            "keys": {cfg.id: cfg.stats for cfg in self._keys.values()},
        }

    def get_stats(self) -> dict:
        total = sum(cfg.stats["total_requests"] for cfg in self._keys.values())
        success = sum(cfg.stats["successes"] for cfg in self._keys.values())

        return {
            "total_requests": total,
            "total_success": success,
            "success_rate": round(success / total * 100 if total > 0 else 0, 1),
            "healthy_keys": self.get_healthy_key_count(),
            "circuit_breaker": "open" if self.is_circuit_open() else "closed",
        }

    def reload_keys(self):
        self._keys.clear()
        self._reset_circuit()
        self._load_keys()
        logger.info(f"Keys reloaded: {len(self._keys)} total, {self.get_healthy_key_count()} healthy")

    def reset_key(self, key_id: str) -> bool:
        for cfg in self._keys.values():
            if cfg.id == key_id:
                cfg._is_available = True
                cfg._failure_count = 0
                cfg._last_failure = None
                self._rebuild_rotation()
                logger.info(f"Key {cfg.id} manually reset")
                return True
        return False

    def reset_all(self):
        for cfg in self._keys.values():
            cfg._is_available = True
            cfg._failure_count = 0
            cfg._last_failure = None
        self._reset_circuit()
        self._rebuild_rotation()
        logger.info("All keys manually reset")


gemini_manager = GeminiClientManager()


# ==============================================================================
# FAST SENTIMENT ANALYSIS
# ==============================================================================

def get_sentiment_polarity(text: str) -> float:
    if USE_VADER:
        scores = vader_analyzer.polarity_scores(text)
        return scores['compound']
    else:
        text_lower = text.lower()
        score = 0.0
        
        for kw in STRONG_POSITIVE_KEYWORDS:
            if kw in text_lower:
                score += 0.3
        for kw in SOFT_POSITIVE_KEYWORDS:
            if kw in text_lower:
                score += 0.1
        for kw in STRONG_NEGATIVE_KEYWORDS:
            if kw in text_lower:
                score -= 0.3
        for kw in SOFT_NEGATIVE_KEYWORDS:
            if kw in text_lower:
                score -= 0.1
        
        return max(-1.0, min(1.0, score))


# ==============================================================================
# SPAM DETECTION
# ==============================================================================

def validate_review_content(review: str) -> tuple[bool, str]:
    review = review.strip()

    if len(review) < MIN_REVIEW_LENGTH:
        return False, f"Review too short (minimum {MIN_REVIEW_LENGTH} characters)"

    letter_ratio = sum(1 for c in review if c.isalpha()) / len(review)
    if letter_ratio < 0.3:
        return False, "Review contains too many non-letter characters"

    if re.search(r'(.)\1{5,}', review):
        return False, "Review contains repeated character patterns"

    caps_ratio = sum(1 for c in review if c.isupper()) / len(review)
    if caps_ratio > 0.8 and len(review) > 20:
        return False, "Review contains excessive capital letters"

    return True, ""


def detect_spam(reviews: list[str]) -> tuple[bool, str]:
    review_counts = Counter(reviews)
    for review, count in review_counts.items():
        if count > SPAM_SAME_REVIEW_LIMIT:
            return True, f"Same review repeated {count} times"

    for i, review1 in enumerate(reviews):
        for review2 in reviews[i+1:]:
            if SequenceMatcher(None, review1.lower(), review2.lower()).ratio() >= SPAM_SIMILARITY_THRESHOLD:
                return True, "Multiple nearly identical reviews detected"

    return False, ""


# ==============================================================================
# SENTIMENT & FEATURES
# ==============================================================================

def classify_sentence_deterministic(sentence: str) -> str:
    text_lower = sentence.lower()
    polarity = get_sentiment_polarity(sentence)

    if polarity > SENTIMENT_POLARITY_THRESHOLD:
        return "positive"
    if polarity < -SENTIMENT_POLARITY_THRESHOLD:
        return "negative"
    if any(kw in text_lower for kw in STRONG_NEGATIVE_KEYWORDS):
        return "negative"
    if any(kw in text_lower for kw in STRONG_POSITIVE_KEYWORDS):
        return "positive"
    if any(kw in text_lower for kw in SOFT_NEGATIVE_KEYWORDS):
        return "negative"
    if any(kw in text_lower for kw in SOFT_POSITIVE_KEYWORDS):
        return "positive"
    return "neutral"


def get_trigger_words(sentence: str, label: str) -> list[str]:
    text_lower = sentence.lower()
    triggers = []

    if label == "positive":
        for kw in STRONG_POSITIVE_KEYWORDS | SOFT_POSITIVE_KEYWORDS:
            if kw in text_lower:
                triggers.append(kw)
    elif label == "negative":
        for kw in STRONG_NEGATIVE_KEYWORDS | SOFT_NEGATIVE_KEYWORDS:
            if kw in text_lower:
                triggers.append(kw)

    return triggers[:5]


def get_classification_confidence(sentence: str, label: str) -> float:
    text_lower = sentence.lower()
    polarity = abs(get_sentiment_polarity(sentence))

    confidence = 0.5 + polarity * 0.3

    if label == "positive":
        confidence += 0.2 if any(kw in text_lower for kw in STRONG_POSITIVE_KEYWORDS) else 0
        confidence += 0.1 if any(kw in text_lower for kw in SOFT_POSITIVE_KEYWORDS) else 0
    elif label == "negative":
        confidence += 0.2 if any(kw in text_lower for kw in STRONG_NEGATIVE_KEYWORDS) else 0
        confidence += 0.1 if any(kw in text_lower for kw in SOFT_NEGATIVE_KEYWORDS) else 0

    return min(1.0, confidence)


def get_feature_weight(feature: str) -> float:
    return FEATURE_WEIGHTS.get(feature, 1.0)


def extract_feature_from_sentence(sentence: str) -> Optional[str]:
    text_lower = sentence.lower()

    for feature, aliases in FEATURE_ALIAS_MAP.items():
        if any(alias in text_lower for alias in aliases):
            return feature

    for anchor in FEATURE_ANCHORS:
        if anchor in text_lower:
            return anchor

    inferred = {
        "performance": {"fast", "slow", "lag", "lagging", "smooth", "speed", "gaming", "hot", "heats"},
        "display": {"bright", "dim", "color", "screen", "resolution", "oled", "lcd", "refresh"},
        "battery": {"drain", "drains", "charge", "charging", "power", "battery life"},
        "camera": {"photo", "video", "selfie", "picture", "zoom", "focus", "portrait"},
        "sound": {"loud", "quiet", "bass", "audio", "speaker", "microphone"},
        "build": {"cheap", "premium", "solid", "flimsy", "glass", "metal", "plastic", "durable"},
        "charging": {"charging", "charge", "charger", "fast charge", "usb", "wireless"},
        "comfort": {"comfortable", "fit", "pain", "heavy", "light", "ergonomic"},
        "connectivity": {"wifi", "bluetooth", "signal", "connection", "range"},
    }

    for feature, patterns in inferred.items():
        if any(p in text_lower for p in patterns):
            return feature

    return None


# ==============================================================================
# TEXT PROCESSING
# ==============================================================================

def clean_clause_start(clause: str) -> str:
    clause = clause.strip()
    for prefix in ("but ", "but", "however ", "however", "although ", "though ", "yet ", "whereas "):
        if clause.lower().startswith(prefix):
            clause = clause[len(prefix):].strip()
            break
    if clause and clause[0].islower():
        clause = clause[0].upper() + clause[1:]
    return clause


def split_into_clauses(text: str) -> list[str]:
    segments = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        clauses = re.split(r"\s+(?:but|however|although|though|yet|whereas|except)\s+", sentence, flags=re.I)
        for clause in clauses:
            cleaned = clean_clause_start(clause).strip(" ,.;")
            if cleaned and len(cleaned) >= MIN_REVIEW_CHARS:
                segments.append(cleaned)
    return segments


def is_valid_fragment(text: str) -> bool:
    if len(text) < MIN_REVIEW_CHARS:
        return False
    words = text.split()
    if len(words) < MIN_REVIEW_WORDS:
        return False
    if words[0].lower() in frozenset({"and", "but", "or", "so", "except"}):
        return False
    if words[-1].lower() in frozenset({"and", "but", "or", "so", "except"}):
        return False
    return extract_feature_from_sentence(text) is not None


def normalize_point(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" -*\t\r\n")
    cleaned = re.sub(r"^\d+[\).\s-]+", "", cleaned)
    letter_count = sum(1 for c in cleaned if c.isalpha())
    uppercase_count = sum(1 for c in cleaned if c.isupper())
    if cleaned.istitle() or (letter_count > 0 and (uppercase_count / letter_count) >= 0.6):
        cleaned = cleaned.lower()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    if len(cleaned) > 120:
        cleaned = cleaned[:117].rstrip() + "..."
    return cleaned


def get_point_signature(text: str) -> str:
    norm = normalize_point(text).lower()
    norm = re.sub(r"[^a-z0-9 ]", "", norm)
    tokens = [t for t in norm.split() if t not in COMPARISON_STOP_WORDS]
    return " ".join(tokens)


def points_semantically_overlap(a: str, b: str) -> bool:
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
    def tokenize(s): return set(re.sub(r"[^a-z0-9 ]", "", s).split())
    tokens_a, tokens_b = tokenize(norm_a), tokenize(norm_b)
    if len(tokens_a) < 2 or len(tokens_b) < 2:
        return False
    overlap = len(tokens_a & tokens_b)
    smaller = min(len(tokens_a), len(tokens_b))
    return overlap >= max(2, math.ceil(smaller * 0.5)) and (overlap / smaller) >= 0.5


def is_useless_point(text: str) -> bool:
    text_lower = text.lower()
    if text_lower.startswith(GENERIC_POINT_MARKERS):
        return True
    return any(phrase in text_lower for phrase in USELESS_POINT_PHRASES)


def exclude_cross_list(items: list[str], excluded: list[str]) -> list[str]:
    return [item for item in items if not any(points_semantically_overlap(item, ex) for ex in excluded if ex)]


def score_point_quality(point: str, label: str) -> float:
    score = 0.0
    text_lower = point.lower()
    feature = extract_feature_from_sentence(point)
    if feature:
        score += 3.0 * get_feature_weight(feature)
    if label == "positive":
        score += 2.0 if any(kw in text_lower for kw in STRONG_POSITIVE_KEYWORDS) else 0
        score += 1.0 if any(kw in text_lower for kw in SOFT_POSITIVE_KEYWORDS) else 0
    elif label == "negative":
        score += 2.0 if any(kw in text_lower for kw in STRONG_NEGATIVE_KEYWORDS) else 0
        score += 1.0 if any(kw in text_lower for kw in SOFT_NEGATIVE_KEYWORDS) else 0
    if any(word in text_lower for word in {"very", "extremely", "really", "quite", "somewhat"}):
        score += 0.5
    word_count = len(point.split())
    score += 1.0 if 4 <= word_count <= 12 else (0.5 if word_count > 12 else 0)
    if is_useless_point(point):
        score -= 5.0
    return max(0.0, score)


def select_best_points(points_a: list[str], points_b: list[str], label: str) -> list[str]:
    all_points = []
    seen = set()
    for point in points_a + points_b:
        sig = get_point_signature(point)
        if sig not in seen:
            seen.add(sig)
            all_points.append((point, score_point_quality(point, label)))
    all_points.sort(key=lambda x: x[1], reverse=True)
    selected = []
    for point, _ in all_points:
        if not any(points_semantically_overlap(point, sp) for sp in selected):
            selected.append(point)
            if len(selected) >= MAX_POINTS:
                break
    return selected


def calculate_feature_scores(reviews: list[str]) -> list[FeatureScore]:
    feature_data = {}
    for review in reviews:
        for clause in split_into_clauses(review):
            if not is_valid_fragment(clause):
                continue
            label = classify_sentence_deterministic(clause)
            feature = extract_feature_from_sentence(clause)
            if not feature:
                continue
            normalized = normalize_point(clause)
            feature_data.setdefault(feature, {"positive": [], "negative": [], "neutral": []})
            if label == "positive":
                feature_data[feature]["positive"].append(normalized)
            elif label == "negative":
                feature_data[feature]["negative"].append(normalized)
            else:
                feature_data[feature]["neutral"].append(normalized)

    scores = []
    for feature, data in feature_data.items():
        pos, neg, neu = len(data["positive"]), len(data["negative"]), len(data["neutral"])
        total = pos + neg + neu
        if total == 0:
            continue
        score = ((pos - neg) / total) * 100
        weight = get_feature_weight(feature)
        scores.append(FeatureScore(
            feature=feature, display_name=feature.capitalize(),
            positive_count=pos, negative_count=neg, neutral_count=neu,
            total_mentions=total, score=round(score, 1), weight=weight,
            weighted_score=round(score * weight, 1),
            examples_positive=data["positive"][:2], examples_negative=data["negative"][:2],
        ))
    scores.sort(key=lambda x: abs(x.weighted_score), reverse=True)
    return scores


def create_explainable_point(text: str, label: str) -> ExplainablePoint:
    return ExplainablePoint(
        text=text,
        feature=extract_feature_from_sentence(text) or "general",
        sentiment=label,
        trigger_words=get_trigger_words(text, label),
        polarity_score=round(get_sentiment_polarity(text), 3),
        confidence=round(get_classification_confidence(text, label), 2),
    )


# ==============================================================================
# RULE-BASED EXTRACTION
# ==============================================================================

def extract_points_rule_based(reviews: list[str]) -> dict[str, list[str]]:
    pros, cons, neutral = [], [], []
    seen = set()
    for review in reviews:
        for clause in split_into_clauses(review):
            if not is_valid_fragment(clause):
                continue
            label = classify_sentence_deterministic(clause)
            normalized = normalize_point(clause)
            sig = get_point_signature(normalized)
            if sig in seen or is_useless_point(normalized):
                continue
            seen.add(sig)
            if label == "positive":
                pros.append(normalized)
            elif label == "negative":
                cons.append(normalized)
            elif "overall" not in normalized.lower() and "mixed" not in normalized.lower():
                neutral.append(normalized)
    return {
        "pros": pros[:MAX_POINTS],
        "cons": cons[:MAX_POINTS],
        "neutral_points": neutral[:MAX_NEUTRAL_POINTS_PER_REVIEW * 2],
    }


def calculate_sentiment_from_clauses(clauses: list[str]) -> dict[str, float | int]:
    counts = {"positive": 0, "neutral": 0, "negative": 0, "total": len(clauses)}
    for clause in clauses:
        label = classify_sentence_deterministic(clause)
        counts[label] += 1
    total = counts["total"] or 1
    return {
        "positive": round(counts["positive"] / total * 100, 2),
        "neutral": round(counts["neutral"] / total * 100, 2),
        "negative": round(counts["negative"] / total * 100, 2),
        "total": counts["total"],
    }


def validate_ai_analysis(ai_analysis: AIAnalysis) -> AIAnalysis:
    def valid_point(p, label):
        return classify_sentence_deterministic(p) == label and extract_feature_from_sentence(p) and not is_useless_point(p)

    validated_pros = [p for p in ai_analysis.pros if valid_point(p, "positive")]
    validated_cons = [c for c in ai_analysis.cons if valid_point(c, "negative")]
    validated_neutral = [
        n for n in ai_analysis.neutral_points
        if classify_sentence_deterministic(n) == "neutral" and not is_useless_point(n)
    ]

    final_pros = exclude_cross_list(validated_pros, validated_cons)
    final_cons = exclude_cross_list(validated_cons, final_pros)

    return AIAnalysis(
        summary=ai_analysis.summary,
        pros=final_pros[:MAX_POINTS],
        cons=final_cons[:MAX_POINTS],
        neutral_points=validated_neutral[:MAX_NEUTRAL_POINTS_PER_REVIEW],
    )


def is_review_related(text: str) -> bool:
    text_lower = text.lower().strip()
    if not text_lower or text_lower.startswith(QUESTION_PREFIXES):
        return False
    if any(word in text_lower for word in PRODUCT_KEYWORDS + SENTIMENT_WORDS):
        return True
    return len(text_lower) > 15 and len(text_lower.split()) >= 3


# ==============================================================================
# AI HELPERS
# ==============================================================================

def resolve_model_name(raw: str) -> str:
    name = raw.strip() or DEFAULT_GEMINI_MODEL
    return LEGACY_MODEL_ALIASES.get(name, name)


def parse_ai_response(raw_text: str) -> AIAnalysis:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()

    def ensure_list(value):
        if value is None: return []
        if isinstance(value, str):
            return [p.strip(" -*\t") for p in re.split(r"[\n,]+", value) if p.strip(" -*\t")]
        if isinstance(value, list):
            return [str(i).strip() for i in value if str(i).strip()]
        return [str(value).strip()] if str(value).strip() else []

    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return AIAnalysis(
                summary=str(payload.get("summary", "")).strip(),
                pros=ensure_list(payload.get("pros")),
                cons=ensure_list(payload.get("cons")),
                neutral_points=ensure_list(payload.get("neutral_points")),
            )
    except (json.JSONDecodeError, Exception):
        pass

    sm = re.search(r"summary\s*:\s*(.+?)(?:\n\s*pros?\s*:|\Z)", cleaned, flags=re.I | re.S)
    pm = re.search(r"pros?\s*:\s*(.+?)(?:\n\s*cons?\s*:|\Z)", cleaned, flags=re.I | re.S)
    cm = re.search(r"cons?\s*:\s*(.+?)(?:\n\s*neutral|\Z)", cleaned, flags=re.I | re.S)
    nm = re.search(r"neutral[_\s-]*points?\s*:\s*(.+)", cleaned, flags=re.I | re.S)

    def extract_list(text):
        if not text: return []
        return [l.strip(" -*\t") for l in text.splitlines() if l.strip() and ":" not in l.lower()]

    return AIAnalysis(
        summary=sm.group(1).strip() if sm else "",
        pros=extract_list(pm.group(1) if pm else ""),
        cons=extract_list(cm.group(1) if cm else ""),
        neutral_points=extract_list(nm.group(1) if nm else ""),
    )


def build_analysis_prompt(reviews: list[str]) -> str:
    return f"""You are analyzing product reviews. Extract ONLY factual insights.

STRICT RULES:
1. Each point MUST mention a product feature
2. Split mixed sentences: "camera good BUT battery bad" → pros: ["camera quality is good"], cons: ["battery performance is poor"]
3. MAX: 2 pros, 2 cons, 1 neutral point
4. NO generic phrases
5. Points CANNOT appear in multiple lists
6. NEVER invent

Return STRICT JSON:
{{
  "summary": "1 sentence verdict",
  "pros": ["specific feature + positive observation"],
  "cons": ["specific feature + negative observation"],
  "neutral_points": ["neutral observation"]
}}

Reviews:
{chr(10).join('- ' + r[:200] for r in reviews[:5])}""".strip()


def build_summary_prompt(pros: list[str], cons: list[str]) -> str:
    return f"""Summarize in 1-2 sentences. Mention specific features.

Pros: {pros[:3] if pros else 'None'}
Cons: {cons[:3] if cons else 'None'}

Return ONLY the summary text:""".strip()


# ==============================================================================
# INTELLIGENT SUMMARY
# ==============================================================================

def build_intelligent_summary(pros: list[str], cons: list[str], sentiment: dict) -> str:
    pos_pct = sentiment.get("positive", 0)
    neg_pct = sentiment.get("negative", 0)
    neu_pct = sentiment.get("neutral", 0)

    def get_features(points, max_count=2):
        features = []
        for p in points:
            f = extract_feature_from_sentence(p)
            if f and f not in features:
                features.append(f)
                if len(features) >= max_count:
                    break
        return features

    pro_feats = get_features(pros, 2)
    con_feats = get_features(cons, 1)

    if pos_pct >= 60:
        tone = "predominantly positive"
    elif neg_pct >= 60:
        tone = "predominantly negative"
    elif pos_pct > neg_pct + 15:
        tone = "mostly positive"
    elif neg_pct > pos_pct + 15:
        tone = "mostly negative"
    elif neu_pct >= 50:
        tone = "mixed"
    else:
        tone = "balanced"

    parts = []

    if tone == "predominantly positive":
        parts.append(f"Reviews are overwhelmingly positive ({pos_pct:.0f}%)")
    elif tone == "predominantly negative":
        parts.append(f"Reviews are largely negative ({neg_pct:.0f}%)")
    elif tone == "mostly positive":
        parts.append(f"Reviews lean positive ({pos_pct:.0f}% positive, {neg_pct:.0f}% negative)")
    elif tone == "mostly negative":
        parts.append(f"Reviews lean negative ({neg_pct:.0f}% negative, {pos_pct:.0f}% positive)")
    elif tone == "mixed":
        parts.append(f"Reviews are mixed ({pos_pct:.0f}% positive, {neg_pct:.0f}% negative)")
    else:
        parts.append(f"Reviews show balanced feedback ({pos_pct:.0f}% positive, {neg_pct:.0f}% negative)")

    if pro_feats:
        if tone in ["predominantly positive", "mostly positive"]:
            parts.append(f"praise for {', '.join(pro_feats[:2])}")
        else:
            parts.append(f"strengths in {', '.join(pro_feats[:2])}")

    if con_feats:
        parts.append(f"concerns about {', '.join(con_feats)}")

    return ". ".join(parts) + "."


# ==============================================================================
# MAIN ANALYSIS (Fully Async)
# ==============================================================================

async def analyze_reviews_complete(
    raw_reviews: list[str],
    api_key: str,
    use_cache: bool = True
) -> tuple[AIAnalysis, dict, list[FeatureScore], bool]:

    cache_key = cache_manager.generate_cache_key(raw_reviews, api_key)
    cached = cache_manager.get(cache_key)

    if cached and use_cache:
        key_preview = hashlib.sha256(api_key.encode()).hexdigest()[:6]
        logger.info(f"Cache HIT for user {key_preview}...")
        return (
            AIAnalysis(**cached["analysis"]),
            cached["sentiment"],
            [FeatureScore(**fs) for fs in cached["feature_scores"]],
            True
        )

    processed = prepare_reviews(raw_reviews)

    if not processed:
        return (
            AIAnalysis(summary="No valid review content found.", pros=[], cons=[], neutral_points=[]),
            {"positive": 0.0, "neutral": 0.0, "negative": 0.0, "total": 0},
            [],
            False
        )

    logger.info("Analyzing %d fragments", len(processed))

    rule_based = extract_points_rule_based(processed)
    sentiment = calculate_sentiment_from_clauses(processed)
    feature_scores = calculate_feature_scores(processed)

    ai_analysis = None
    ai_summary = None

    if gemini_manager.has_keys():
        try:
            model_name = resolve_model_name(os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))

            ai_analysis = await gemini_manager.analyze_reviews(processed, model_name)

            if ai_analysis:
                ai_analysis = validate_ai_analysis(ai_analysis)
                ai_summary = await gemini_manager.generate_summary(
                    ai_analysis.pros,
                    ai_analysis.cons,
                    model_name
                )

        except Exception as exc:
            logger.warning(f"AI enhancement failed: {exc}")

    if ai_analysis and (ai_analysis.pros or ai_analysis.cons):
        final_pros = select_best_points(ai_analysis.pros, rule_based["pros"], "positive")
        final_cons = select_best_points(ai_analysis.cons, rule_based["cons"], "negative")
        final_pros = exclude_cross_list(final_pros, final_cons)
        final_cons = exclude_cross_list(final_cons, final_pros)

        if ai_summary and len(ai_summary.split()) >= 5:
            final_summary = ai_summary
        else:
            final_summary = build_intelligent_summary(final_pros, final_cons, sentiment)

        final_analysis = AIAnalysis(
            summary=final_summary,
            pros=final_pros[:MAX_POINTS],
            cons=final_cons[:MAX_POINTS],
            neutral_points=rule_based["neutral_points"][:MAX_NEUTRAL_POINTS_PER_REVIEW]
        )
    else:
        final_summary = build_intelligent_summary(rule_based["pros"], rule_based["cons"], sentiment)
        final_analysis = AIAnalysis(
            summary=final_summary,
            pros=rule_based["pros"][:MAX_POINTS],
            cons=rule_based["cons"][:MAX_POINTS],
            neutral_points=rule_based["neutral_points"][:MAX_NEUTRAL_POINTS_PER_REVIEW]
        )

    cache_manager.set(cache_key, {
        "analysis": final_analysis.model_dump(),
        "sentiment": sentiment,
        "feature_scores": [fs.model_dump() for fs in feature_scores]
    })

    return final_analysis, sentiment, feature_scores, False


def prepare_reviews(raw_reviews: list[str]) -> list[str]:
    prepared = []
    for raw in raw_reviews:
        for fragment in split_into_clauses(raw):
            if is_valid_fragment(fragment):
                prepared.append(fragment)
    return prepared[:MAX_REVIEWS_TO_ANALYZE]


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
# RATE LIMIT HANDLER
# ==============================================================================

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Too many requests. Please slow down."})


# ==============================================================================
# MAIN ENDPOINT (with Analytics Logging)
# ==============================================================================

@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
async def analyze_reviews(
    request: Request,
    payload: ReviewRequest,
    x_api_key: Optional[str] = Header(None)
) -> AnalyzeResponse:
    api_key = await verify_api_key(x_api_key)
    client_ip = request.client.host if request.client else "unknown"
    key_preview = hashlib.sha256(api_key.encode()).hexdigest()[:6]
    identifier = f"{key_preview}:{client_ip}"

    # ✅ Analytics: Log request start
    request_start_time = time.time()
    
    is_blocked, remaining = cooldown_protection.is_blocked(api_key, client_ip)
    if is_blocked:
        # ✅ Analytics: Log blocked request
        logger.info(
            f"ANALYTICS | blocked | user={key_preview} | ip={client_ip} | "
            f"reviews={len(payload.reviews)} | blocked_remaining={remaining}s"
        )
        raise HTTPException(status_code=429, detail=f"Account blocked. Try again in {remaining}s.")

    is_limited, reason = scalable_limiter.is_rate_limited(identifier)
    if is_limited:
        cooldown_protection.record_abuse(api_key, client_ip, reason)
        # ✅ Analytics: Log rate limited
        logger.info(
            f"ANALYTICS | rate_limited | user={key_preview} | ip={client_ip} | "
            f"reviews={len(payload.reviews)} | reason={reason}"
        )
        raise HTTPException(status_code=429, detail=reason)

    if len(payload.reviews) > MAX_REVIEWS_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Too many reviews. Max {MAX_REVIEWS_PER_REQUEST}.")

    is_spam, spam_reason = detect_spam(payload.reviews)
    if is_spam:
        cooldown_protection.record_abuse(api_key, client_ip, spam_reason)
        logger.info(
            f"ANALYTICS | spam | user={key_preview} | ip={client_ip} | "
            f"reviews={len(payload.reviews)} | reason={spam_reason}"
        )
        raise HTTPException(status_code=400, detail=f"Spam detected: {spam_reason}")

    for i, review in enumerate(payload.reviews):
        is_valid, error = validate_review_content(review)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Review {i+1}: {error}")

    for review in payload.reviews:
        if not is_review_related(review):
            raise HTTPException(status_code=400, detail="Input does not appear to be product reviews.")

    scalable_limiter.record_request(identifier)

    # ✅ Analytics: Log successful request start
    logger.info(
        f"ANALYTICS | request_start | user={key_preview} | ip={client_ip} | "
        f"reviews={len(payload.reviews)}"
    )

    analysis, sentiment, feature_scores, from_cache = await analyze_reviews_complete(
        payload.reviews, api_key, ENABLE_CACHE
    )

    score, confidence = calculate_score_and_confidence(sentiment)
    
    # ✅ Analytics: Log completed request
    duration_ms = (time.time() - request_start_time) * 1000
    logger.info(
        f"ANALYTICS | request_complete | user={key_preview} | ip={client_ip} | "
        f"reviews={len(payload.reviews)} | cached={from_cache} | "
        f"score={score:.2f} | confidence={confidence:.1f}% | "
        f"duration_ms={duration_ms:.0f} | positive={sentiment['positive']:.1f}% | "
        f"negative={sentiment['negative']:.1f}%"
    )

    return AnalyzeResponse(
        summary=analysis.summary,
        pros=analysis.pros,
        cons=analysis.cons,
        neutral_points=analysis.neutral_points,
        sentiment=SentimentBreakdown(
            positive=sentiment["positive"],
            neutral=sentiment["neutral"],
            negative=sentiment["negative"],
            total=sentiment["total"]
        ),
        score=score,
        confidence=confidence,
        explained_pros=[create_explainable_point(p, "positive") for p in analysis.pros],
        explained_cons=[create_explainable_point(c, "negative") for c in analysis.cons],
        feature_scores=feature_scores,
        cached=from_cache,
    )


# ==============================================================================
# ADMIN ENDPOINTS
# ==============================================================================

@app.get("/stats")
async def get_stats(x_api_key: Optional[str] = Header(None)) -> dict:
    await verify_api_key(x_api_key)
    return {
        "cache": cache_manager.get_stats(),
        "redis_connected": redis_client is not None,
        "api_keys_configured": len(VALID_API_KEYS)
    }


@app.post("/cache/clear")
async def clear_cache(x_api_key: Optional[str] = Header(None)) -> dict:
    await verify_api_key(x_api_key)
    cache_manager.lru_cache.cache.clear()
    
    if redis_client:
        deleted = cache_manager.clear_redis_cache()
        return {"status": "cache cleared", "redis_keys_deleted": deleted}
    
    return {"status": "cache cleared"}


@app.get("/admin/gemini-health")
async def get_gemini_health(x_api_key: Optional[str] = Header(None)) -> dict:
    await verify_api_key(x_api_key)
    return gemini_manager.get_health()


@app.get("/admin/gemini-stats")
async def get_gemini_stats(x_api_key: Optional[str] = Header(None)) -> dict:
    await verify_api_key(x_api_key)
    return gemini_manager.get_stats()


@app.post("/admin/gemini-reload")
async def reload_gemini(x_api_key: Optional[str] = Header(None)) -> dict:
    await verify_api_key(x_api_key)
    gemini_manager.reload_keys()
    return {"status": "reloaded", **gemini_manager.get_health()}


@app.post("/admin/gemini-reset/{key_id}")
async def reset_gemini_key(key_id: str, x_api_key: Optional[str] = Header(None)) -> dict:
    await verify_api_key(x_api_key)
    if gemini_manager.reset_key(key_id):
        return {"status": f"key {key_id} reset"}
    return {"error": f"key {key_id} not found"}, 404


@app.post("/admin/gemini-reset-all")
async def reset_all_keys(x_api_key: Optional[str] = Header(None)) -> dict:
    await verify_api_key(x_api_key)
    gemini_manager.reset_all()
    return {"status": "all keys reset"}


# ==============================================================================
# ROUTES
# ==============================================================================

@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "AI Product Review Aggregator API v9.4", "docs": "/docs"}

@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
