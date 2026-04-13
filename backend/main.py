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
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict

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

# Cache settings
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1000"))
MAX_REDIS_CACHE_SIZE = int(os.getenv("MAX_REDIS_CACHE_SIZE", "10000"))

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


# ==============================================================================
# KEYWORD SETS
# ==============================================================================

STRONG_NEGATIVE = frozenset({
    "bad", "poor", "worst", "waste", "overheat", "lag", "drain", "heats",
    "fails", "failure", "crash", "buggy", "terrible", "horrible", "awful", "broken",
})

STRONG_POSITIVE = frozenset({
    "excellent", "great", "smooth", "easy", "premium", "bright",
    "sharp", "clean", "love", "best", "amazing",
})

SOFT_NEGATIVE = frozenset({
    "slow", "weak", "issue", "problem", "expensive", "overpriced", "noisy", "hot",
})

SOFT_POSITIVE = frozenset({
    "good", "nice", "fast", "clear", "lightweight", "sleek", "fine", "okay", "decent",
})

ALL_POSITIVE = STRONG_POSITIVE | SOFT_POSITIVE
ALL_NEGATIVE = STRONG_NEGATIVE | SOFT_NEGATIVE

FEATURE_ALIAS_MAP = {
    "battery": {"battery", "backup", "drain", "mah", "power"},
    "camera": {"camera", "photo", "video", "lens", "focus", "zoom", "selfie"},
    "performance": {"performance", "lag", "slow", "speed", "processor", "ram", "gaming"},
    "design": {"design", "look", "style", "color"},
    "display": {"display", "screen", "brightness", "touch", "oled", "lcd"},
    "sound": {"sound", "audio", "speaker", "volume", "bass"},
    "charging": {"charging", "charge", "charger", "fast charge"},
    "build": {"build", "quality", "durable", "material", "plastic", "metal"},
    "price": {"price", "cost", "expensive", "value"},
    "software": {"software", "ui", "update", "app", "os"},
    "support": {"support", "service", "warranty"},
    "comfort": {"comfort", "fit", "pain", "ear", "heavy", "light"},
    "connectivity": {"wifi", "bluetooth", "signal", "network"},
}

VALID_REVIEW_WORDS = frozenset({
    "good", "bad", "great", "nice", "okay", "fine", "poor", "excellent",
    "quality", "product", "works", "love", "hate", "fast", "slow",
    "buy", "use", "using", "recommend", "rating", "star", "review",
    "best", "worst", "terrible", "amazing", "average", "phone", "laptop",
    "battery", "performance", "camera", "screen", "display", "sound",
    "design", "build", "price", "software", "charge", "works", "working",
})


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


class FeatureScore(BaseModel):
    feature: str
    display_name: str
    positive_count: int
    negative_count: int
    total_mentions: int
    score: float


class AnalyzeResponse(BaseModel):
    summary: str
    pros: list[str]
    cons: list[str]
    neutral_points: list[str] = Field(default_factory=list)
    sentiment: SentimentBreakdown
    score: float
    confidence: float
    cached: bool = False
    warnings: list[str] = Field(default_factory=list)
    explained_pros: list[ExplainablePoint] | None = None
    explained_cons: list[ExplainablePoint] | None = None
    feature_scores: list[FeatureScore] | None = None


# ==============================================================================
# FASTAPI APP SETUP
# ==============================================================================

app = FastAPI(
    title="AI Product Review Aggregator API",
    description="Production-ready elite-level system",
    version="18.0",
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
        logger.warning("No API keys configured - running in open mode!")
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
    
    @app.get("/metrics")
    async def metrics():
        return JSONResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    USE_PROMETHEUS = True
    
except ImportError:
    USE_PROMETHEUS = False
    
    class NoOpCounter:
        def labels(self, **kwargs): return self
        def inc(self, n=1): pass
    class NoOpHistogram:
        def labels(self, **kwargs):
            class ctx:
                def __enter__(self): pass
                def __exit__(self, *args): pass
            return ctx()
    class NoOpGauge:
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
            "hit_rate": f"{hit_rate:.1}%",
            "entries": self.size,
            "max_size": self.max_size
        }


class CacheManager:
    def __init__(self):
        self.lru_cache = LRUCache(max_size=MAX_CACHE_SIZE)
        self.redis_count = 0
        self.cache_set_count = 0

    def generate_cache_key(self, reviews: list[str]) -> str:
        sorted_reviews = sorted(reviews)
        reviews_hash = hashlib.sha256(json.dumps(sorted_reviews, sort_keys=True).encode()).hexdigest()
        return f"review_cache:{reviews_hash[:32]}"

    def get(self, key: str) -> Optional[dict]:
        if not ENABLE_CACHE:
            return None

        if redis_client and self.redis_count > MAX_REDIS_CACHE_SIZE:
            return self.lru_cache.get(key)

        if redis_client:
            try:
                data = redis_client.get(key)
                if data:
                    return json.loads(data)
            except Exception:
                pass

        return self.lru_cache.get(key)

    def set(self, key: str, data: dict, ttl: int = None):
        if not ENABLE_CACHE:
            return

        ttl = ttl or CACHE_TTL_SECONDS
        data_str = json.dumps(data)

        if redis_client:
            try:
                if self.redis_count <= MAX_REDIS_CACHE_SIZE:
                    redis_client.setex(key, ttl, data_str)
                    self.redis_count += 1
                    self.cache_set_count += 1
                    return
            except Exception:
                pass

        self.lru_cache.set(key, data, ttl)

    def sync_redis_count(self):
        if not redis_client:
            return
        
        try:
            cursor = 0
            count = 0
            pattern = "review_cache:*"
            
            while True:
                cursor, keys = redis_client.scan(cursor=cursor, match=pattern, count=100)
                count += len(keys)
                if cursor == 0:
                    break
            
            self.redis_count = count
        except Exception:
            pass

    def clear_redis_cache(self) -> int:
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
        except Exception:
            pass

        self.redis_count = 0
        return deleted


cache_manager = CacheManager()


# ==============================================================================
# RATE LIMITER
# ==============================================================================

class ScalableRateLimiter:
    def __init__(self):
        self._requests = {}

    def record_request(self, identifier: str):
        now = time.time()

        if identifier in self._requests:
            self._requests[identifier] = [t for t in self._requests[identifier] if now - t < 3600]

        if redis_client:
            try:
                key = f"rate:requests:{identifier}"
                pipe = redis_client.pipeline()
                pipe.lpush(key, now)
                pipe.ltrim(key, 0, 999)
                pipe.expire(key, 3600)
                pipe.execute()
                return
            except Exception:
                pass

        if identifier not in self._requests:
            self._requests[identifier] = []
        self._requests[identifier].append(now)

    def get_request_count(self, identifier: str, window_seconds: int = 60) -> int:
        now = time.time()

        if redis_client:
            try:
                key = f"rate:requests:{identifier}"
                cutoff = now - window_seconds
                redis_client.ltrim(key, 0, 999)
                requests = redis_client.lrange(key, 0, -1)
                return sum(1 for r in requests if float(r) > cutoff)
            except Exception:
                pass

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

# ✅ FIX: Task tracking with automatic cleanup
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
        logger.info(f"GeminiClientManager initialized: {len(self._keys)} key(s)")

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
        rate_limit_indicators = {"429", "rate limit", "quota", "too many requests"}
        if any(p in msg for p in rate_limit_indicators):
            return True, True
        transient_indicators = {"500", "502", "503", "504", "timeout", "connection", "unavailable"}
        if any(p in msg for p in transient_indicators):
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

    async def analyze_reviews(self, reviews: list[str], model_name: str = DEFAULT_GEMINI_MODEL) -> Optional[dict]:
        def _call(client):
            return client.models.generate_content(
                model=model_name,
                contents=build_analysis_prompt(reviews),
                config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json"),
            )

        result, error = await self._execute_with_retry(_call)
        if error or not result:
            return None
        return parse_ai_response(result.text or "")

    async def generate_summary(self, pros: list[str], cons: list[str], model_name: str = DEFAULT_GEMINI_MODEL) -> Optional[str]:
        def _call(client):
            return client.models.generate_content(
                model=model_name,
                contents=build_summary_prompt(pros, cons),
                config=types.GenerateContentConfig(temperature=0.3, response_mime_type="text/plain"),
            )

        result, error = await self._execute_with_retry(_call)
        if error or not result:
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


gemini_manager = GeminiClientManager()


# ==============================================================================
# SENTIMENT & TEXT PROCESSING
# ==============================================================================

def get_sentiment_polarity(text: str) -> float:
    if USE_VADER:
        return vader_analyzer.polarity_scores(text)['compound']
    else:
        text_lower = text.lower()
        score = 0.0
        for kw in STRONG_POSITIVE:
            if kw in text_lower: score += 0.3
        for kw in SOFT_POSITIVE:
            if kw in text_lower: score += 0.1
        for kw in STRONG_NEGATIVE:
            if kw in text_lower: score -= 0.3
        for kw in SOFT_NEGATIVE:
            if kw in text_lower: score -= 0.1
        return max(-1.0, min(1.0, score))


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


def is_review_related(text: str) -> bool:
    text_lower = text.lower().strip()
    if not text_lower or len(text_lower) < 8:
        return False
    if len(text_lower) > 50 and len(text_lower.split()) >= 4:
        return True
    if any(word in text_lower for word in VALID_REVIEW_WORDS):
        return True
    if re.search(r'\d+', text_lower):
        return True
    if len(text_lower.split()) >= 3:
        return True
    return False


def split_into_clauses(text: str) -> list[str]:
    segments = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        clauses = re.split(r"\s+(?:but|however|although|though|yet|whereas)\s+", sentence, flags=re.I)
        for clause in clauses:
            clause = clause.strip()
            for prefix in ("but ", "however ", "although ", "though "):
                if clause.lower().startswith(prefix):
                    clause = clause[len(prefix):].strip()
                    break
            if clause and clause[0].islower():
                clause = clause[0].upper() + clause[1:]
            clause = clause.strip(" ,.;")
            if clause and len(clause) >= 10:
                segments.append(clause)
    return segments


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


@lru_cache(maxsize=1000)
def extract_feature_cached(sentence: str) -> Optional[str]:
    text_lower = sentence.lower()

    for feature, aliases in FEATURE_ALIAS_MAP.items():
        if any(alias in text_lower for alias in aliases):
            return feature

    inferred = {
        "performance": {"fast", "slow", "lag", "speed", "gaming"},
        "display": {"bright", "screen", "color", "oled", "lcd"},
        "battery": {"drain", "charge", "power"},
        "camera": {"photo", "video", "picture", "zoom"},
        "sound": {"loud", "audio", "speaker"},
        "build": {"cheap", "premium", "solid", "durable"},
    }

    for feature, patterns in inferred.items():
        if any(p in text_lower for p in patterns):
            return feature

    return None


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


def extract_points(clauses: list[str]) -> dict[str, list[str]]:
    pros, cons, neutral = [], [], []
    seen = set()

    for clause in clauses:
        label = classify_sentence(clause)
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
        "neutral_points": neutral[:MAX_NEUTRAL_POINTS],
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


def calculate_feature_scores(clauses: list[str]) -> list[FeatureScore]:
    feature_data = {}

    for clause in clauses:
        if not is_valid_fragment(clause):
            continue

        label = classify_sentence(clause)
        feature = extract_feature_cached(clause)
        if not feature:
            continue

        normalized = normalize_point(clause)
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
            display_name=feature.capitalize(),
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
                "pros": [p for p in pros if isinstance(p, str) and len(p.strip()) >= 5],
                "cons": [c for c in cons if isinstance(c, str) and len(c.strip()) >= 5],
                "neutral_points": [n for n in payload.get("neutral_points", []) if isinstance(n, str) and len(n.strip()) >= 5],
            }
    except (json.JSONDecodeError, Exception):
        pass

    sm = re.search(r"summary\s*:\s*(.+?)(?:\n\s*pros?\s*:|\Z)", cleaned, flags=re.I | re.S)
    pm = re.search(r"pros?\s*:\s*(.+?)(?:\n\s*cons?\s*:|\Z)", cleaned, flags=re.I | re.S)
    cm = re.search(r"cons?\s*:\s*(.+?)(?:\n\s*neutral|\Z)", cleaned, flags=re.I | re.S)

    def extract_list(text):
        if not text:
            return []
        return [l.strip() for l in text.splitlines() if l.strip() and ":" not in l.lower() and len(l.strip()) >= 5]

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


def build_analysis_prompt(reviews: list[str]) -> str:
    return f"""You are analyzing product reviews. Extract ONLY factual insights.

STRICT RULES:
1. Each point MUST mention a product feature
2. Split mixed sentences: "camera good BUT battery bad"
3. MAX: 2 pros, 2 cons, 1 neutral point
4. NO generic phrases
5. NEVER invent

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


def build_summary(pros: list[str], cons: list[str], sentiment: dict) -> str:
    pos_pct = sentiment.get("positive", 0)
    neg_pct = sentiment.get("negative", 0)
    neu_pct = sentiment.get("neutral", 0)

    if pos_pct >= 60:
        tone = "positive"
    elif neg_pct >= 60:
        tone = "negative"
    elif pos_pct > neg_pct + 15:
        tone = "positive"
    elif neg_pct > pos_pct + 15:
        tone = "negative"
    elif neu_pct >= 50:
        tone = "mixed"
    else:
        tone = "balanced"

    def get_features(points, max_count=2):
        features = []
        for p in points:
            f = extract_feature_cached(p)
            if f and f not in features:
                features.append(f)
                if len(features) >= max_count:
                    break
        return features

    pro_feats = get_features(pros, 2)
    con_feats = get_features(cons, 1)

    parts = []

    if tone == "positive":
        parts.append(f"Reviews are predominantly positive ({pos_pct:.0f}%)")
        if pro_feats:
            parts.append(f"with praise for {', '.join(pro_feats[:2])}")
    elif tone == "negative":
        parts.append(f"Reviews are largely negative ({neg_pct:.0f}%)")
        if con_feats:
            parts.append(f"with concerns about {', '.join(con_feats)}")
    elif tone == "mixed":
        parts.append(f"Reviews are mixed ({pos_pct:.0f}% positive, {neg_pct:.0f}% negative)")
    else:
        parts.append(f"Reviews show balanced feedback ({pos_pct:.0f}% positive, {neg_pct:.0f}% negative)")

    return ". ".join(parts) + "."


def select_best_points(points_a: list[str], points_b: list[str], label: str) -> list[str]:
    seen = set()
    all_points = []

    for point in points_a + points_b:
        sig = get_point_signature(point)
        if sig in seen:
            continue

        text_lower = point.lower()
        score = 0.0

        if extract_feature_cached(point):
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

    selected = []
    for point, _ in all_points:
        if not any(points_overlap(point, sp) for sp in selected):
            selected.append(point)
            if len(selected) >= MAX_POINTS:
                break

    return selected


# ==============================================================================
# WORKER PROCESSOR
# ==============================================================================

async def process_analysis_task(reviews: list[str], detailed: bool) -> tuple:
    warnings = []
    start_time = time.time()

    cache_key = cache_manager.generate_cache_key(reviews)
    cached = cache_manager.get(cache_key)

    if cached:
        CACHE_HITS.inc()
        return (
            cached["analysis"],
            cached["sentiment"],
            [FeatureScore(**fs) for fs in cached.get("feature_scores", [])],
            True,
            []
        )

    clauses = prepare_and_split(reviews)

    if not clauses:
        return (
            {"summary": "No valid review content found.", "pros": [], "cons": [], "neutral_points": []},
            {"positive": 0.0, "neutral": 0.0, "negative": 0.0, "total": 0},
            [],
            False,
            []
        )

    rule_based = extract_points(clauses)
    sentiment = calculate_sentiment(clauses)

    ai_result = None
    ai_summary = None

    if gemini_manager.has_keys():
        try:
            model_name = resolve_model_name(os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))

            async with gemini_semaphore:
                request_tracker.active_gemini += 1
                try:
                    ai_result = await asyncio.wait_for(
                        gemini_manager.analyze_reviews(clauses, model_name),
                        timeout=GEMINI_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    warnings.append("⚠️ AI enhancement timed out, using rule-based analysis.")
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

        except Exception:
            warnings.append("⚠️ AI enhancement unavailable, using rule-based analysis.")

    if ai_result and (ai_result.get("pros") or ai_result.get("cons")):
        final_pros = select_best_points(ai_result.get("pros", []), rule_based["pros"], "positive")
        final_cons = select_best_points(ai_result.get("cons", []), rule_based["cons"], "negative")

        final_pros = [p for p in final_pros if not any(points_overlap(p, c) for c in final_cons)]
        final_cons = [c for c in final_cons if not any(points_overlap(c, p) for p in final_pros)]

        if ai_summary and len(ai_summary.split()) >= 5:
            final_summary = ai_summary
        else:
            final_summary = build_summary(final_pros, final_cons, sentiment)

        final_analysis = {
            "summary": final_summary,
            "pros": final_pros[:MAX_POINTS],
            "cons": final_cons[:MAX_POINTS],
            "neutral_points": rule_based["neutral_points"],
        }
    else:
        final_summary = build_summary(rule_based["pros"], rule_based["cons"], sentiment)
        final_analysis = {
            "summary": final_summary,
            "pros": rule_based["pros"],
            "cons": rule_based["cons"],
            "neutral_points": rule_based["neutral_points"],
        }

    cache_data = {
        "analysis": final_analysis,
        "sentiment": sentiment,
    }

    if detailed:
        cache_data["feature_scores"] = [fs.model_dump() for fs in calculate_feature_scores(clauses)]

    cache_manager.set(cache_key, cache_data)
    feature_scores = calculate_feature_scores(clauses) if detailed else []

    return final_analysis, sentiment, feature_scores, False, warnings


# ==============================================================================
# ✅ MULTIPLE WORKER LOOPS
# ==============================================================================

async def worker_loop(worker_id: int):
    """Worker that processes tasks from queue."""
    logger.info(f"Worker {worker_id} started")
    
    while True:
        try:
            task_data = await asyncio.wait_for(WORKER_QUEUE.get(), timeout=1.0)
            
            task_id, reviews, detailed, future = task_data
            
            try:
                result = await process_analysis_task(reviews, detailed)
                
                if not future.done():
                    future.set_result(result)
                    
            except Exception as exc:
                logger.error(f"Worker {worker_id} task {task_id} error: {exc}")
                if not future.done():
                    future.set_exception(exc)
            finally:
                request_tracker.queue_depth = WORKER_QUEUE.qsize()
                WORKER_QUEUE.task_done()
                
                # ✅ FIX: Cleanup task from tracking
                with TASK_RESULTS_LOCK:
                    TASK_RESULTS.pop(task_id, None)
                
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.error(f"Worker {worker_id} loop error: {e}")
            await asyncio.sleep(1)


async def redis_sync_task():
    """Periodic sync of Redis cache count."""
    while True:
        await asyncio.sleep(300)
        try:
            cache_manager.sync_redis_count()
        except Exception:
            pass


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
# ✅ MAIN ENDPOINT
# ==============================================================================

@app.post("/analyze-raw", response_model=AnalyzeResponse)
async def analyze_raw_text(
    request: Request,
    payload: RawAnalyzeRequest,
    x_api_key: Optional[str] = Header(None)
) -> AnalyzeResponse:
    endpoint = "/analyze-raw"
    start_time = time.time()

    request_tracker.active_http += 1

    try:
        api_key = await verify_api_key(x_api_key)
        user_id = hashlib.sha256(api_key.encode()).hexdigest()[:6]

        # Rate limit check
        is_limited, reason = scalable_limiter.is_rate_limited(user_id)
        if is_limited:
            REQUEST_COUNT.labels(endpoint=endpoint, status="rate_limited").inc()
            raise HTTPException(status_code=429, detail=reason)

        if len(payload.raw_text) > MAX_INPUT_SIZE:
            raise HTTPException(status_code=400, detail=f"Input too large. Max {MAX_INPUT_SIZE} chars.")

        reviews = parse_raw_input(payload.raw_text)

        if not reviews:
            raise HTTPException(status_code=400, detail="⚠️ Could not extract reviews.")

        review_related_count = sum(1 for r in reviews if is_review_related(r))
        warnings = []

        if review_related_count == 0:
            warnings.append("⚠️ Input may not be well-structured. Attempting analysis anyway.")

        scalable_limiter.record_request(user_id)

        detailed = request.query_params.get("detailed", "false").lower() == "true"

        # ✅ Submit to worker queue
        task_id = generate_task_id()
        
        # ✅ FIX: Use create_future() instead of Future()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        # Register for tracking
        with TASK_RESULTS_LOCK:
            TASK_RESULTS[task_id] = future
        
        try:
            WORKER_QUEUE.put_nowait((task_id, reviews, detailed, future))
            request_tracker.queue_depth = WORKER_QUEUE.qsize()
        except asyncio.QueueFull:
            REQUEST_COUNT.labels(endpoint=endpoint, status="queue_full").inc()
            raise HTTPException(status_code=503, detail="Server busy. Try again later.")

        # ✅ Wait for worker result
        with REQUEST_LATENCY.labels(endpoint=endpoint).time():
            try:
                analysis, sentiment, feature_scores, from_cache, ai_warnings = await asyncio.wait_for(future, timeout=30.0)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Processing timeout. Try again.")

        warnings.extend(ai_warnings)

        score, confidence = calculate_score_and_confidence(sentiment)

        response = AnalyzeResponse(
            summary=analysis["summary"],
            pros=analysis["pros"],
            cons=analysis["cons"],
            neutral_points=analysis.get("neutral_points", []),
            sentiment=SentimentBreakdown(
                positive=sentiment["positive"],
                neutral=sentiment["neutral"],
                negative=sentiment["negative"],
                total=sentiment["total"]
            ),
            score=score,
            confidence=confidence,
            cached=from_cache,
            warnings=warnings if warnings else None,
        )

        if detailed:
            response.explained_pros = [
                ExplainablePoint(
                    text=p,
                    feature=extract_feature_cached(p) or "general",
                    sentiment="positive",
                    polarity_score=round(get_sentiment_polarity(p), 3),
                )
                for p in analysis["pros"]
            ]
            response.explained_cons = [
                ExplainablePoint(
                    text=c,
                    feature=extract_feature_cached(c) or "general",
                    sentiment="negative",
                    polarity_score=round(get_sentiment_polarity(c), 3),
                )
                for c in analysis["cons"]
            ]
            response.feature_scores = feature_scores

        REQUEST_COUNT.labels(endpoint=endpoint, status="success").inc()
        logger.info(f"ANALYTICS | complete | user={user_id} | time={time.time() - start_time:.3f}s | cached={from_cache}")

        return response

    except HTTPException:
        raise
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, status="error").inc()
        logger.error(f"ANALYTICS | error | {str(exc)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        request_tracker.active_http -= 1


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_reviews(
    request: Request,
    payload: ReviewRequest,
    x_api_key: Optional[str] = Header(None)
) -> AnalyzeResponse:
    raw_text = "\n".join(payload.reviews)
    return await analyze_raw_text(request, RawAnalyzeRequest(raw_text=raw_text), x_api_key)


# ==============================================================================
# ADMIN ENDPOINTS
# ==============================================================================

@app.get("/stats")
async def get_stats(x_api_key: Optional[str] = Header(None)) -> dict:
    await verify_api_key(x_api_key)
    return {
        "cache": cache_manager.lru_cache.get_stats(),
        "redis_connected": redis_client is not None,
        "redis_count": cache_manager.redis_count,
    }


@app.post("/cache/clear")
async def clear_cache(x_api_key: Optional[str] = Header(None)) -> dict:
    await verify_api_key(x_api_key)
    cache_manager.lru_cache.cache.clear()
    cache_manager.lru_cache.size = 0

    if redis_client:
        deleted = cache_manager.clear_redis_cache()
        return {"status": "cache cleared", "redis_keys_deleted": deleted}

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
            "redis_entries": cache_manager.redis_count,
            "memory_entries": cache_manager.lru_cache.size,
        },
        "active_requests": {
            "http": request_tracker.active_http,
            "gemini": request_tracker.active_gemini,
        },
    }


# ==============================================================================
# ROUTES
# ==============================================================================

@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "message": "AI Product Review Aggregator API v18.0",
        "docs": "/docs",
        "primary_endpoint": "/analyze-raw",
    }

@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


# ==============================================================================
# ✅ STARTUP EVENT
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Start multiple worker loops for parallel processing."""
    # ✅ FIX: Spawn multiple workers
    for i in range(WORKER_POOL_SIZE):
        asyncio.create_task(worker_loop(i))
    
    # Start Redis sync task
    asyncio.create_task(redis_sync_task())
    
    logger.info(f"Started {WORKER_POOL_SIZE} worker loops")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
