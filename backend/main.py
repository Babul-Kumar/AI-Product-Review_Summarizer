import json
import logging
import math
import os
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError, field_validator
from textblob import TextBlob

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.DEBUG))
logger = logging.getLogger(__name__)

PLACEHOLDER_API_KEYS = {"your_gemini_api_key_here", "replace_with_real_key"}
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
LEGACY_MODEL_ALIASES = {
    "gemini-pro": DEFAULT_GEMINI_MODEL,
}
MAX_POINTS = 6
MAX_POINTS_PER_REVIEW = 2
MAX_NEUTRAL_POINTS_PER_REVIEW = 1
MAX_REVIEWS = 100
MAX_REVIEWS_TO_ANALYZE = 10
MIN_REVIEW_CHARS = 15
MIN_REVIEW_WORDS = 3
MIN_COMMA_SPLIT_WORDS = 10
MIN_AND_SPLIT_WORDS = 6
SENTIMENT_POLARITY_THRESHOLD = 0.1
MIN_POINT_CHARS = 8

NEUTRAL_CUES = (
    "okay",
    "ok",
    "average",
    "decent",
    "fine",
    "nothing special",
    "acceptable",
    "as expected",
)
POSITIVE_CUES = (
    "excellent",
    "great",
    "smooth",
    "easy",
    "premium",
    "bright",
    "loud",
    "sharp",
    "clean",
    "fast",
)
NEGATIVE_CUES = (
    "overpriced",
    "price is too high",
    "too high",
    "expensive",
    "slow",
    "slow charging",
    "heats up",
    "heating",
    "weak",
    "poor",
    "lag",
    "issue",
    "problem",
    "terrible",
    "bad",
)
STRONG_NEGATIVE_KEYWORDS = (
    "not",
    "bad",
    "poor",
    "worst",
    "waste",
    "slow",
    "issue",
    "problem",
    "crash",
    "bug",
    "buggy",
    "overheat",
    "overheating",
    "lag",
    "lagging",
    "drain",
    "drains",
    "heats",
    "heating",
    "fails",
    "failure",
)
COMPARISON_STOP_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "for",
    "from",
    "in",
    "is",
    "of",
    "the",
    "to",
    "very",
    "with",
}
GENERIC_POINT_MARKERS = (
    "no major",
    "no clear",
    "no complaints",
    "no complaint",
    "no cons",
    "no negatives",
    "no issues",
    "none",
    "not mentioned",
    "n/a",
)
USELESS_POINT_PHRASES = (
    "no major recurring strengths",
    "no major recurring complaints",
    "no complaints mentioned",
    "no complaint mentioned",
    "no complaints",
    "no issues",
)
LOW_QUALITY_PHRASES = (
    "good product",
    "great product",
    "nice product",
    "bad product",
    "average product",
    "works fine",
    "it is good",
    "it is bad",
)
PRODUCT_KEYWORDS = (
    "battery",
    "camera",
    "performance",
    "design",
    "quality",
    "price",
    "screen",
    "display",
    "speaker",
    "software",
    "support",
    "charging",
    "build",
    "setup",
)
SENTIMENT_WORDS = (
    "good",
    "bad",
    "poor",
    "excellent",
    "slow",
    "fast",
    "great",
    "amazing",
    "terrible",
    "buggy",
    "average",
)
QUESTION_PREFIXES = (
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "tell me",
    "explain",
    "define",
)
FEATURE_SIGNAL_KEYWORDS = tuple(
    dict.fromkeys(
        PRODUCT_KEYWORDS
        + (
            "lag",
            "lagging",
            "overheating",
            "heating",
            "thermal",
            "speed",
            "performance",
            "audio",
            "sound",
            "mic",
            "network",
            "signal",
            "ui",
            "ux",
            "update",
            "processor",
            "ram",
            "storage",
            "gaming",
            "touch",
            "brightness",
            "refresh",
            "focus",
            "zoom",
            "stabilization",
            "charge",
            "charging",
            "ear",
            "comfort",
            "fit",
        )
    )
)
FEATURE_ANCHOR_KEYWORDS = (
    "battery",
    "camera",
    "performance",
    "design",
    "display",
    "sound",
    "charging",
    "build",
    "price",
    "software",
    "support",
    "comfort",
    "fit",
)
FEATURE_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "battery": ("battery", "backup", "drain", "drains", "mah"),
    "camera": ("camera", "photo", "photos", "video", "portrait", "lens", "focus", "zoom"),
    "performance": ("performance", "lag", "lagging", "slow", "speed", "processor", "ram", "gaming"),
    "design": ("design", "look", "looks", "style"),
    "display": ("display", "screen", "brightness", "touch", "refresh"),
    "sound": ("sound", "audio", "speaker", "mic", "microphone"),
    "charging": ("charging", "charge", "charger"),
    "build": ("build", "quality", "durable", "durability", "material"),
    "price": ("price", "cost", "expensive", "overpriced", "value"),
    "software": ("software", "ui", "ux", "update", "app"),
    "support": ("support", "service", "warranty"),
    "comfort": ("comfort", "comfortable", "ear", "pain", "fit", "lightweight"),
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ReviewRequest(BaseModel):
    reviews: list[str] = Field(..., min_length=1)

    @field_validator("reviews")
    @classmethod
    def clean_reviews(cls, reviews: list[str]) -> list[str]:
        cleaned_reviews = [review.strip() for review in reviews if review and review.strip()]
        if not cleaned_reviews:
            raise ValueError("Please provide at least one non-empty review.")
        if len(cleaned_reviews) > MAX_REVIEWS:
            raise ValueError(f"Please provide no more than {MAX_REVIEWS} reviews at a time.")
        return cleaned_reviews


class SentimentBreakdown(BaseModel):
    positive: float
    neutral: float
    negative: float
    total: int


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


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Product Review Aggregator API")

# Allow all origins to support dynamic frontend deployments (e.g., Vercel).
# allow_credentials must stay False when allow_origins=["*"]; setting it to
# True with a wildcard origin is a security misconfiguration.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "AI Product Review API",
    }


# ---------------------------------------------------------------------------
# Scope detection
# ---------------------------------------------------------------------------

def is_review_related_rule_based(text: str) -> bool | None:
    """Fast rule-based scope check.

    Returns True/False when confident, None when uncertain (triggers AI check).
    ``str.startswith`` accepts a tuple of prefixes natively — this checks
    whether *any* of the QUESTION_PREFIXES match the beginning of the text.
    """
    text_lower = text.lower().strip()
    if not text_lower:
        return False

    if text_lower.startswith(QUESTION_PREFIXES):
        return False

    if any(word in text_lower for word in PRODUCT_KEYWORDS + SENTIMENT_WORDS):
        return True

    if len(text_lower) > 15 and len(text_lower.split()) >= 3:
        return True

    return None


def is_review_related(text: str) -> bool:
    """Hybrid scope check: rule-based first, AI fallback for edge cases.

    FIX (Bug 3): When both rule-based result is None *and* the AI call fails,
    the original code returned True unconditionally — allowing off-topic input
    through. Changed to return False (fail-closed) on AI failure so that
    ambiguous content is rejected rather than silently accepted.
    """
    rule_based_result = is_review_related_rule_based(text)

    if rule_based_result is not None:
        logger.info("Scope check: rule=%s, ai=None", rule_based_result)
        return rule_based_result

    try:
        ai_result = is_review_related_ai(text)
        logger.info("Scope check: rule=None, ai=%s", ai_result)
        return ai_result
    except Exception as exc:
        logger.warning("Hybrid scope detection AI fallback failed: %s", exc)
        # Fail closed: ambiguous input is rejected when AI is unreachable.
        return False


# ---------------------------------------------------------------------------
# Sentiment helpers
# ---------------------------------------------------------------------------

def classify_sentiment(text: str) -> tuple[str, float]:
    polarity = TextBlob(text).sentiment.polarity
    lowered_text = text.lower()

    if any(keyword in lowered_text for keyword in STRONG_NEGATIVE_KEYWORDS):
        return "negative", polarity

    if polarity > SENTIMENT_POLARITY_THRESHOLD:
        return "positive", polarity

    if polarity < -SENTIMENT_POLARITY_THRESHOLD:
        return "negative", polarity

    return "neutral", polarity


def calculate_sentiment_from_reviews(
    reviews: list[str],
    ai_analysis: AIAnalysis | None = None,
) -> dict[str, int | float]:
    positive_count = 0
    neutral_count = 0
    negative_count = 0
    positive_weight = 0.0
    neutral_weight = 0.0
    negative_weight = 0.0
    source = "review-weighted"

    for review in reviews:
        sentiment_label, polarity = classify_sentiment(review)
        lowered_review = review.lower()
        base_weight = 1.0 + min(1.5, abs(polarity) * 2)

        if sentiment_label == "positive":
            cue_hits = sum(1 for cue in POSITIVE_CUES if cue in lowered_review)
            weight = base_weight + min(0.8, cue_hits * 0.12)
            positive_count += 1
            positive_weight += weight
        elif sentiment_label == "negative":
            cue_hits = sum(1 for cue in STRONG_NEGATIVE_KEYWORDS if cue in lowered_review)
            weight = base_weight + min(1.2, cue_hits * 0.2)
            negative_count += 1
            negative_weight += weight
        else:
            neutral_count += 1
            neutral_weight += 1.0

    total_reviews = positive_count + neutral_count + negative_count

    # Fallback when no valid review fragments were retained.
    if total_reviews == 0 and ai_analysis is not None:
        positive_count = len(ai_analysis.pros)
        negative_count = len(ai_analysis.cons)
        neutral_count = len(ai_analysis.neutral_points)
        positive_weight = float(positive_count)
        negative_weight = float(negative_count)
        neutral_weight = float(neutral_count)
        total_reviews = positive_count + neutral_count + negative_count
        source = "insight-fallback"

    logger.info(
        "Sentiment derived from %s: %s positive, %s negative, %s neutral.",
        source,
        positive_count,
        negative_count,
        neutral_count,
    )

    if total_reviews == 0:
        return {
            "positive": 0.0,
            "neutral": 0.0,
            "negative": 0.0,
            "total": 0,
            "score": 0.0,
            "confidence": 0.0,
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 0,
        }

    positive_percentage, neutral_percentage, negative_percentage = calculate_percentages(
        positive_weight,
        neutral_weight,
        negative_weight,
        positive_weight + neutral_weight + negative_weight,
    )
    score = calculate_score(
        positive_weight,
        negative_weight,
        positive_weight + neutral_weight + negative_weight,
    )
    confidence = calculate_confidence(
        positive_count,
        neutral_count,
        negative_count,
        total_reviews,
    )

    return {
        "positive": positive_percentage,
        "neutral": neutral_percentage,
        "negative": negative_percentage,
        "total": total_reviews,
        "score": score,
        "confidence": confidence,
        "positive_count": positive_count,
        "neutral_count": neutral_count,
        "negative_count": negative_count,
    }


# ---------------------------------------------------------------------------
# Text splitting / fragment helpers
# ---------------------------------------------------------------------------

def split_sentences(review: str) -> list[str]:
    segments: list[str] = []

    for sentence in re.split(r"(?<=[.!?])\s+", review):
        if not sentence.strip():
            continue

        clauses = re.split(r"\s+(?:but|however|although|though|yet)\s+", sentence, flags=re.I)
        for clause in clauses:
            cleaned_clause = clause.strip(" ,.;")
            if cleaned_clause:
                segments.append(cleaned_clause)

    return segments


def split_reviews(text: str) -> list[str]:
    if not text.strip():
        return []

    split_candidates: list[str] = []
    sentence_chunks = re.split(r"\n+|(?<=[.!?])\s+", text)

    for sentence_chunk in sentence_chunks:
        cleaned_sentence = re.sub(r"\s+", " ", sentence_chunk).strip(" ,;:-")
        if not cleaned_sentence:
            continue

        contrast_chunks = re.split(
            r"\s+(?:but|however|although|though|yet|whereas)\s+",
            cleaned_sentence,
            flags=re.I,
        )

        for contrast_chunk in contrast_chunks:
            cleaned_clause = re.sub(r"\s+", " ", contrast_chunk).strip(" ,;:-")
            if not cleaned_clause:
                continue

            comma_chunks = [cleaned_clause]
            has_multi_feature_signal = sum(
                1
                for feature in FEATURE_ANCHOR_KEYWORDS
                if re.search(rf"\b{re.escape(feature)}\b", cleaned_clause, flags=re.I)
            ) >= 2
            if (
                "," in cleaned_clause
                and len(cleaned_clause.split()) >= MIN_COMMA_SPLIT_WORDS
                and has_multi_feature_signal
            ):
                comma_chunks = re.split(r"\s*,\s*", cleaned_clause)

            for chunk in comma_chunks:
                normalized_chunk = re.sub(r"\s+", " ", chunk).strip(" ,;:-")
                if is_valid_review_fragment(normalized_chunk):
                    split_candidates.append(normalized_chunk)

    return list(dict.fromkeys(split_candidates))


def is_valid_review_fragment(text: str) -> bool:
    cleaned_text = re.sub(r"\s+", " ", text).strip(" ,;:-")
    if len(cleaned_text) < MIN_REVIEW_CHARS:
        return False
    words = cleaned_text.split()
    if len(words) < MIN_REVIEW_WORDS:
        return False
    if words[0].lower() in {"and", "but", "or", "so"}:
        return False
    if words[-1].lower() in {"and", "but", "or", "so"}:
        return False
    return bool(re.search(r"[a-zA-Z]", cleaned_text))


# ---------------------------------------------------------------------------
# Point normalisation / deduplication
# ---------------------------------------------------------------------------

def normalize_point(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" -*\t\r\n")
    cleaned = re.sub(r"^\d+[\).\s-]+", "", cleaned)
    cleaned = normalize_display_text(cleaned)
    if len(cleaned) > 120:
        cleaned = f"{cleaned[:117].rstrip()}..."
    return cleaned


def normalize_display_text(text: str) -> str:
    """Normalise casing to sentence case.

    Lowercases text when the majority of alphabetic characters are uppercase
    (e.g. all-caps user input), then capitalises the first character.
    Uses a 60 % threshold which handles fully-uppercased strings and title-
    cased strings; mixed-case text (e.g. "Great OLED display") is left as-is
    because the uppercase ratio stays below the threshold.
    """
    cleaned = text.strip()
    if not cleaned:
        return ""

    letter_count = sum(1 for char in cleaned if char.isalpha())
    uppercase_count = sum(1 for char in cleaned if char.isupper())
    should_sentence_case = cleaned.istitle() or (
        letter_count > 0 and (uppercase_count / letter_count) >= 0.6
    )

    if should_sentence_case:
        cleaned = cleaned.lower()

    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned


def point_signature(text: str) -> str:
    normalized_text = normalize_point(text).lower().strip()
    normalized_text = re.sub(r"[^a-z0-9 ]", "", normalized_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
    tokens = [token for token in normalized_text.split() if token not in COMPARISON_STOP_WORDS]
    return " ".join(tokens)


def point_signature_tokens(text: str) -> list[str]:
    return point_signature(text).split()


def points_overlap(existing: str, candidate: str) -> bool:
    normalized_existing = normalize_point(existing).lower()
    normalized_candidate = normalize_point(candidate).lower()

    if not normalized_existing or not normalized_candidate:
        return False

    if (
        normalized_existing.startswith(GENERIC_POINT_MARKERS)
        or normalized_candidate.startswith(GENERIC_POINT_MARKERS)
    ):
        return normalized_existing == normalized_candidate

    if normalized_existing == normalized_candidate:
        return True

    if normalized_existing in normalized_candidate or normalized_candidate in normalized_existing:
        return True

    if SequenceMatcher(None, normalized_existing, normalized_candidate).ratio() >= 0.84:
        return True

    existing_tokens = point_signature_tokens(existing)
    candidate_tokens = point_signature_tokens(candidate)
    if len(existing_tokens) < 2 or len(candidate_tokens) < 2:
        return False

    existing_set = set(existing_tokens)
    candidate_set = set(candidate_tokens)
    if existing_set <= candidate_set or candidate_set <= existing_set:
        return True

    overlap_count = len(existing_set & candidate_set)
    smaller_set_size = min(len(existing_set), len(candidate_set))
    minimum_overlap = max(2, math.ceil(smaller_set_size * 0.5))
    return overlap_count >= minimum_overlap and (overlap_count / smaller_set_size) >= 0.5


def should_replace_point(existing: str, candidate: str) -> bool:
    existing_lower = normalize_point(existing).lower()
    candidate_lower = normalize_point(candidate).lower()
    existing_feature_hits = sum(1 for keyword in FEATURE_SIGNAL_KEYWORDS if keyword in existing_lower)
    candidate_feature_hits = sum(1 for keyword in FEATURE_SIGNAL_KEYWORDS if keyword in candidate_lower)
    existing_score = (existing_feature_hits, len(point_signature_tokens(existing)), len(existing))
    candidate_score = (candidate_feature_hits, len(point_signature_tokens(candidate)), len(candidate))
    return candidate_score > existing_score


def deduplicate_semantic(points: list[str]) -> list[str]:
    """Deduplicate points using semantic overlap detection.

    FIX (Bug 2): The original implementation had a redundant second
    ``any(points_overlap(...))`` check after the inner loop. When the inner
    loop found an overlap and set ``should_skip = True`` we'd already
    ``continue``, so the second check only ran on the non-overlapping branch
    — where it would always pass, making it dead code. Worse, if the inner
    loop ran to completion without finding an overlap, a later-inserted item
    could still overlap with the new candidate because the second check only
    covers items already in ``result`` at that moment.

    Replaced with a single linear scan that finds the first overlapping item,
    replaces it if the candidate is better, and otherwise skips the candidate.
    """
    result: list[str] = []

    for point in points:
        candidate = normalize_point(point)
        if not candidate:
            continue

        overlapping_index = next(
            (i for i, existing in enumerate(result) if points_overlap(existing, candidate)),
            None,
        )

        if overlapping_index is not None:
            if should_replace_point(result[overlapping_index], candidate):
                result[overlapping_index] = candidate
            # Either way, do not append a duplicate.
        else:
            result.append(candidate)

    return result


def is_useless_point(text: str) -> bool:
    normalized_text = normalize_point(text).lower()
    if not normalized_text:
        return True

    if normalized_text.startswith(GENERIC_POINT_MARKERS):
        return True

    return any(phrase in normalized_text for phrase in USELESS_POINT_PHRASES)


def has_feature_anchor(text: str) -> bool:
    lowered_text = text.lower()

    for aliases in FEATURE_ALIAS_MAP.values():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", lowered_text):
                return True

    return False


def is_low_quality_point(text: str) -> bool:
    normalized_text = normalize_point(text)
    lowered_text = normalized_text.lower()

    if len(normalized_text) < MIN_POINT_CHARS:
        return True

    if lowered_text in LOW_QUALITY_PHRASES:
        return True

    if any(lowered_text.startswith(phrase) for phrase in LOW_QUALITY_PHRASES):
        return True

    if has_feature_anchor(lowered_text):
        return False

    word_count = len(lowered_text.split())
    has_sentiment_cue = any(cue in lowered_text for cue in POSITIVE_CUES + NEGATIVE_CUES + NEUTRAL_CUES)
    return not (word_count >= 4 and len(lowered_text) >= 18 and has_sentiment_cue)


def _build_clean_point_list(items: list[str]) -> list[str]:
    """Shared cleaning pipeline used by both public helpers below."""
    items = deduplicate_semantic(list(dict.fromkeys(items)))
    cleaned_points: list[str] = []
    seen: set[str] = set()

    for item in items:
        cleaned_item = normalize_point(item)
        if len(cleaned_item) < 4:
            continue
        if is_useless_point(cleaned_item):
            continue
        if is_low_quality_point(cleaned_item):
            continue

        point_key = cleaned_item.lower()
        if point_key in seen:
            continue

        seen.add(point_key)
        cleaned_points.append(cleaned_item)

    return cleaned_points[:MAX_POINTS]


def clean_point_list(items: list[str], fallback_message: str = "") -> list[str]:
    """Clean and deduplicate a required point list.

    FIX (Bug 1 + duplicate code): The original version accepted a
    ``fallback_message`` parameter but immediately discarded it
    (``_ = fallback_message``), always returning ``[]`` for empty lists.
    Call-sites expected the fallback string to appear when no valid points
    survived cleaning.

    FIX (duplicate code): ``clean_optional_point_list`` was identical to this
    function minus the unused parameter. Both are now implemented via the
    shared ``_build_clean_point_list`` helper; ``clean_optional_point_list``
    is kept as a thin wrapper for backward compatibility.

    FIX (Bug 5 interaction): Fallback injection now happens *after* cleaning
    so the fallback string is never passed into the cleaning pipeline where
    ``is_useless_point`` would strip it out.
    """
    result = _build_clean_point_list(items)
    if not result and fallback_message:
        return [fallback_message]
    return result


def clean_optional_point_list(items: list[str]) -> list[str]:
    """Clean and deduplicate an optional point list (no fallback required).

    Thin wrapper around the shared pipeline kept for call-site compatibility.
    """
    return _build_clean_point_list(items)


def filter_generic_neutral_points(points: list[str]) -> list[str]:
    return [
        point
        for point in points
        if "overall" not in point.lower()
        and "mixed" not in point.lower()
        and not is_useless_point(point)
    ]


def exclude_existing_points(items: list[str], excluded_items: list[str]) -> list[str]:
    filtered_items: list[str] = []

    for item in items:
        if any(points_overlap(item, excluded_item) for excluded_item in excluded_items if excluded_item):
            continue
        filtered_items.append(item)

    return filtered_items


def extract_points(
    reviews: list[str],
    target_label: str,
) -> list[str]:
    """Extract sentiment-labelled sentences from reviews.

    FIX (Bug 5): The original function accepted a ``fallback_message``
    parameter and returned ``[fallback_message]`` when no points were found.
    That fallback string then entered ``clean_point_list`` which ran it
    through ``is_useless_point`` — which matched it against
    ``USELESS_POINT_PHRASES`` and stripped it out, making the fallback
    completely ineffective.

    The fallback responsibility has been moved to ``clean_point_list`` which
    applies it *after* cleaning.  This function now simply returns ``[]``
    when nothing is found.
    """
    points: list[str] = []
    seen: set[str] = set()

    for review in reviews:
        for sentence in split_sentences(review):
            cleaned_sentence = normalize_point(sentence)
            if len(cleaned_sentence) < 12:
                continue

            sentiment_label, _ = classify_sentiment(cleaned_sentence)
            if target_label == "neutral" and sentiment_label != "neutral":
                continue

            point_key = cleaned_sentence.lower()
            cue_match = False
            if target_label == "negative":
                cue_match = any(cue in point_key for cue in NEGATIVE_CUES)
            elif target_label == "positive":
                cue_match = any(cue in point_key for cue in POSITIVE_CUES)
            elif target_label == "neutral":
                cue_match = any(cue in point_key for cue in NEUTRAL_CUES)

            if (sentiment_label == target_label or cue_match) and point_key not in seen:
                points.append(cleaned_sentence)
                seen.add(point_key)

            if len(points) == MAX_POINTS:
                return points

    return points


# ---------------------------------------------------------------------------
# Fallback analysis (no AI)
# ---------------------------------------------------------------------------

def build_fallback_analysis(reviews: list[str], sentiment: dict[str, int | float]) -> AIAnalysis:
    positive_score = sentiment["positive"]
    neutral_score = sentiment["neutral"]
    negative_score = sentiment["negative"]

    if neutral_score >= max(positive_score, negative_score):
        tone = "mostly neutral or mixed"
    elif positive_score >= negative_score + 15:
        tone = "mostly positive"
    elif negative_score >= positive_score + 15:
        tone = "mostly negative"
    else:
        tone = "mixed"

    # FIX (Bug 5): extract_points no longer injects a fallback internally;
    # clean_point_list receives it and applies it after cleaning.
    pros = clean_point_list(
        extract_points(reviews, "positive"),
        "No major recurring strengths were mentioned.",
    )
    cons = clean_point_list(
        extract_points(reviews, "negative"),
        "No major recurring complaints were mentioned.",
    )

    neutral_points = clean_optional_point_list(extract_points(reviews, "neutral"))
    neutral_points = exclude_existing_points(neutral_points, pros + cons)
    neutral_points = filter_generic_neutral_points(neutral_points)

    summary = (
        f"Overall feedback is {tone}. "
        f"Users highlight strengths like {pros[0] if pros else 'various features'} "
        f"but report issues such as {cons[0] if cons else 'some concerns'}."
    )

    return AIAnalysis(
        summary=summary,
        pros=pros,
        cons=cons,
        neutral_points=neutral_points,
    )


# ---------------------------------------------------------------------------
# Model / client helpers
# ---------------------------------------------------------------------------

def resolve_model_name(raw_model_name: str) -> str:
    cleaned_model_name = raw_model_name.strip() or DEFAULT_GEMINI_MODEL

    if cleaned_model_name in LEGACY_MODEL_ALIASES:
        resolved_model_name = LEGACY_MODEL_ALIASES[cleaned_model_name]
        logger.warning(
            "The model '%s' is legacy. Using '%s' instead.",
            cleaned_model_name,
            resolved_model_name,
        )
        return resolved_model_name

    return cleaned_model_name


def is_review_related_ai(text: str) -> bool:
    """AI-based scope classifier.

    Note: creates its own ``genai.Client`` instance. If scope checks become
    frequent, consider injecting a shared client to avoid repeated
    instantiation overhead.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key in PLACEHOLDER_API_KEYS:
        return False

    model_name = resolve_model_name(os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
    prompt = f"""
Classify the following text:

Is this a product review or user experience related to a product?

Text: {json.dumps(text)}

Answer ONLY "yes" or "no"
""".strip()

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="text/plain",
        ),
    )
    answer = (response.text or "").strip().lower()

    if answer.startswith("yes"):
        return True

    if answer.startswith("no"):
        return False

    # Unexpected response — treat as non-review to be safe.
    return False


# ---------------------------------------------------------------------------
# AI analysis normalisation helpers
# ---------------------------------------------------------------------------

def normalize_analysis(ai_analysis: AIAnalysis) -> AIAnalysis:
    summary = ai_analysis.summary.strip() or "No summary was returned."
    pros = clean_point_list(ai_analysis.pros, "No major recurring strengths were mentioned.")
    cons = clean_point_list(ai_analysis.cons, "No major recurring complaints were mentioned.")
    neutral_points = clean_optional_point_list(ai_analysis.neutral_points)
    neutral_points = exclude_existing_points(neutral_points, pros + cons)
    neutral_points = filter_generic_neutral_points(neutral_points)

    return AIAnalysis(
        summary=summary,
        pros=pros,
        cons=cons,
        neutral_points=neutral_points,
    )


def normalize_analysis_without_fallback(ai_analysis: AIAnalysis) -> AIAnalysis:
    summary = ai_analysis.summary.strip()
    pros = deduplicate_points_with_set(ai_analysis.pros)
    cons = deduplicate_points_with_set(ai_analysis.cons)
    neutral_points = deduplicate_points_with_set(ai_analysis.neutral_points)
    neutral_points = exclude_existing_points(neutral_points, pros + cons)
    neutral_points = filter_generic_neutral_points(neutral_points)

    return AIAnalysis(
        summary=summary,
        pros=pros,
        cons=cons,
        neutral_points=neutral_points,
    )


def log_analysis_details(ai_analysis: AIAnalysis) -> None:
    logger.debug("Merged pros count: %s", len(ai_analysis.pros))
    logger.debug("Merged cons count: %s", len(ai_analysis.cons))
    logger.debug("Merged neutral points count: %s", len(ai_analysis.neutral_points))


def merge_analysis_sources(primary: AIAnalysis, secondary: AIAnalysis) -> AIAnalysis:
    merged_pros = deduplicate_semantic(primary.pros + secondary.pros)
    merged_cons = deduplicate_semantic(primary.cons + secondary.cons)
    merged_neutral_points = deduplicate_semantic(
        primary.neutral_points + secondary.neutral_points,
    )

    merged_cons = exclude_existing_points(merged_cons, [])
    merged_neutral_points = exclude_existing_points(merged_neutral_points, merged_pros + merged_cons)

    pros = clean_point_list(merged_pros, "No major recurring strengths were mentioned.")
    cons = clean_point_list(merged_cons, "No major recurring complaints were mentioned.")
    neutral_points = clean_optional_point_list(merged_neutral_points)
    neutral_points = exclude_existing_points(neutral_points, pros + cons)
    neutral_points = filter_generic_neutral_points(neutral_points)

    return AIAnalysis(
        summary=primary.summary.strip() or secondary.summary,
        pros=pros,
        cons=cons,
        neutral_points=neutral_points,
    )


# ---------------------------------------------------------------------------
# AI response parsing
# ---------------------------------------------------------------------------

def parse_ai_response(raw_text: str) -> AIAnalysis:
    cleaned_text = raw_text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned_text, flags=re.DOTALL).strip()

    def ensure_point_list(value: object) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            split_values = re.split(r"[\n,]+", value)
            return [item.strip(" -*\t") for item in split_values if item.strip(" -*\t")]

        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]

        return [str(value).strip()] if str(value).strip() else []

    try:
        parsed_payload = json.loads(cleaned_text)
        if isinstance(parsed_payload, dict):
            return AIAnalysis(
                summary=str(parsed_payload.get("summary", "")).strip(),
                pros=ensure_point_list(parsed_payload.get("pros")),
                cons=ensure_point_list(parsed_payload.get("cons")),
                neutral_points=ensure_point_list(parsed_payload.get("neutral_points")),
            )

        raise ValueError("Gemini JSON payload must be an object.")
    except (json.JSONDecodeError, ValidationError, ValueError):
        summary_match = re.search(
            r"summary\s*:\s*(.+?)(?:\n\s*pros?\s*:|\Z)", cleaned_text, flags=re.I | re.S
        )
        pros_match = re.search(
            r"pros?\s*:\s*(.+?)(?:\n\s*cons?\s*:|\Z)", cleaned_text, flags=re.I | re.S
        )
        cons_match = re.search(
            r"cons?\s*:\s*(.+?)(?:\n\s*neutral[_\s-]*points?\s*:|\Z)",
            cleaned_text,
            flags=re.I | re.S,
        )
        neutral_match = re.search(
            r"neutral[_\s-]*points?\s*:\s*(.+)",
            cleaned_text,
            flags=re.I | re.S,
        )

        def extract_section_points(section_text: str | None) -> list[str]:
            if not section_text:
                return []

            lines = [line.strip(" -*\t") for line in section_text.splitlines() if line.strip()]
            return [line for line in lines if ":" not in line.lower()]

        summary = summary_match.group(1).strip() if summary_match else cleaned_text
        pros = extract_section_points(pros_match.group(1) if pros_match else None)
        cons = extract_section_points(cons_match.group(1) if cons_match else None)
        neutral_points = extract_section_points(neutral_match.group(1) if neutral_match else None)
        return AIAnalysis(
            summary=summary,
            pros=pros,
            cons=cons,
            neutral_points=neutral_points,
        )


# ---------------------------------------------------------------------------
# Review preparation
# ---------------------------------------------------------------------------

def prepare_reviews(raw_reviews: list[str]) -> list[str]:
    prepared_reviews: list[str] = []

    for raw_review in raw_reviews:
        for split_review in split_reviews(raw_review):
            if is_valid_review_fragment(split_review):
                prepared_reviews.append(split_review)

    if len(prepared_reviews) > MAX_REVIEWS_TO_ANALYZE:
        logger.info(
            "Received %s split reviews; limiting Gemini analysis to first %s.",
            len(prepared_reviews),
            MAX_REVIEWS_TO_ANALYZE,
        )

    return prepared_reviews[:MAX_REVIEWS_TO_ANALYZE]


def deduplicate_points_with_set(points: list[str]) -> list[str]:
    normalized_points: list[str] = []

    for point in points:
        normalized_point = normalize_point(point)
        if not normalized_point:
            continue
        if is_useless_point(normalized_point):
            continue
        if is_low_quality_point(normalized_point):
            continue
        normalized_points.append(normalized_point)

    unique_points = list(dict.fromkeys(normalized_points))
    return deduplicate_semantic(unique_points)


# ---------------------------------------------------------------------------
# Feature anchoring / aggregation
# ---------------------------------------------------------------------------

def extract_feature_anchor(point: str) -> str:
    lowered_point = point.lower()

    for canonical_feature, aliases in FEATURE_ALIAS_MAP.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", lowered_point):
                return canonical_feature

    for keyword in FEATURE_ANCHOR_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", lowered_point):
            return keyword

    for keyword in FEATURE_SIGNAL_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", lowered_point):
            return keyword

    return "other"


def select_representative_points(points: list[str], limit: int) -> list[str]:
    filtered_points: list[str] = []

    for point in points:
        normalized_point = normalize_point(point)
        if not normalized_point:
            continue
        if is_useless_point(normalized_point):
            continue
        if is_low_quality_point(normalized_point):
            continue
        filtered_points.append(normalized_point)

    if not filtered_points or limit <= 0:
        return []

    point_counts = Counter(filtered_points)
    ranked_points = sorted(
        point_counts,
        key=lambda item: (
            point_counts[item],
            len(point_signature_tokens(item)),
            len(item),
        ),
        reverse=True,
    )

    selected_points: list[str] = []
    for point in ranked_points:
        if any(points_overlap(point, existing) for existing in selected_points):
            continue
        selected_points.append(point)
        if len(selected_points) == limit:
            break

    return selected_points


def aggregate_review_analyses(review_analyses: list[AIAnalysis]) -> AIAnalysis:
    feature_map: dict[str, dict[str, list[str]]] = {}

    for analysis in review_analyses:
        for point in analysis.pros:
            feature = extract_feature_anchor(point)
            if feature == "other":
                continue
            feature_map.setdefault(feature, {"pros": [], "cons": [], "neutral": []})
            feature_map[feature]["pros"].append(point)

        for point in analysis.cons:
            feature = extract_feature_anchor(point)
            if feature == "other":
                continue
            feature_map.setdefault(feature, {"pros": [], "cons": [], "neutral": []})
            feature_map[feature]["cons"].append(point)

        for point in analysis.neutral_points:
            feature = extract_feature_anchor(point)
            if feature == "other":
                continue
            feature_map.setdefault(feature, {"pros": [], "cons": [], "neutral": []})
            feature_map[feature]["neutral"].append(point)

    final_pros: list[str] = []
    final_cons: list[str] = []
    final_neutral_points: list[str] = []

    for feature_data in feature_map.values():
        final_pros.extend(select_representative_points(feature_data["pros"], limit=2))
        final_cons.extend(select_representative_points(feature_data["cons"], limit=2))
        final_neutral_points.extend(select_representative_points(feature_data["neutral"], limit=1))

    pros = deduplicate_points_with_set(final_pros)[:MAX_POINTS]
    cons = deduplicate_points_with_set(final_cons)[:MAX_POINTS]
    neutral_points = deduplicate_points_with_set(final_neutral_points)[:MAX_POINTS]
    neutral_points = exclude_existing_points(neutral_points, pros + cons)
    neutral_points = filter_generic_neutral_points(neutral_points)

    return AIAnalysis(
        summary="Aggregated insights based on key product features.",
        pros=pros,
        cons=cons,
        neutral_points=neutral_points,
    )


# ---------------------------------------------------------------------------
# Gemini per-review analysis
# ---------------------------------------------------------------------------

def build_single_review_prompt(review: str) -> str:
    return f"""
You are a STRICT product review analyzer.

Your job:
Extract ONLY factual product insights.

STRICT RULES:
1. Every point MUST mention a product feature:
   (battery, camera, performance, design, display, sound, charging, build, price)

2. DO NOT output generic phrases like:
   - "good product"
   - "nice experience"
   - "works well"

3. Split mixed sentences:
   Example:
   "camera is good but battery is bad"
   -> pros: ["camera quality is good"]
   -> cons: ["battery performance is poor"]

4. NEVER invent information.

5. MAX LIMIT:
   - 2 pros
   - 2 cons
   - 1 neutral

6. If no real feature -> return empty list

Return STRICT JSON ONLY:
{{
  "summary": "1 short sentence",
  "pros": [],
  "cons": [],
  "neutral_points": []
}}

Review:
{review}
""".strip()


def limit_per_review_analysis_points(ai_analysis: AIAnalysis) -> AIAnalysis:
    return AIAnalysis(
        summary=ai_analysis.summary.strip(),
        pros=ai_analysis.pros[:MAX_POINTS_PER_REVIEW],
        cons=ai_analysis.cons[:MAX_POINTS_PER_REVIEW],
        neutral_points=ai_analysis.neutral_points[:MAX_NEUTRAL_POINTS_PER_REVIEW],
    )


def request_review_analysis(client: genai.Client, model_name: str, review: str) -> AIAnalysis:
    response = client.models.generate_content(
        model=model_name,
        contents=build_single_review_prompt(review),
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=AIAnalysis,
        ),
    )
    raw_response_text = (response.text or "").strip()
    if raw_response_text:
        logger.debug("Gemini single-review raw response: %s", raw_response_text)

    if getattr(response, "parsed", None):
        parsed = response.parsed
        validated = parsed if isinstance(parsed, AIAnalysis) else AIAnalysis.model_validate(parsed)
        normalized_analysis = normalize_analysis_without_fallback(validated)
        return limit_per_review_analysis_points(normalized_analysis)

    if raw_response_text:
        parsed_response = parse_ai_response(raw_response_text)
        normalized_analysis = normalize_analysis_without_fallback(parsed_response)
        return limit_per_review_analysis_points(normalized_analysis)

    raise ValueError("Gemini returned an empty single-review response.")


def generate_final_summary(
    client: genai.Client,
    model_name: str,
    pros: list[str],
    cons: list[str],
    fallback_summary: str,
) -> str:
    """Generate a final human-readable summary from aggregated pros/cons.

    FIX (Warning): The original function had no error handling around the
    Gemini call. Any network error or API failure would propagate up to the
    outer ``except Exception`` in ``get_ai_analysis``, discarding all
    per-review analysis already computed and returning the local fallback.
    Now catches all exceptions and returns ``fallback_summary`` gracefully.
    """
    if not pros and not cons:
        return fallback_summary or "No summary was returned."

    prompt = f"""
Generate a final summary based on these pros and cons.

Pros:
{chr(10).join(f"- {point}" for point in pros) or "- None"}

Cons:
{chr(10).join(f"- {point}" for point in cons) or "- None"}

Return only a concise 3-4 sentence summary.
""".strip()

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="text/plain",
            ),
        )
        summary = (response.text or "").strip()
        return summary or fallback_summary or "No summary was returned."
    except Exception as exc:
        logger.warning("Failed to generate final summary from Gemini: %s", exc)
        return fallback_summary or "No summary was returned."


# ---------------------------------------------------------------------------
# Main AI analysis orchestrator
# ---------------------------------------------------------------------------

def get_ai_analysis(reviews: list[str]) -> AIAnalysis:
    processed_reviews = prepare_reviews(reviews)
    if not processed_reviews:
        return AIAnalysis(
            summary="No summary was returned.",
            pros=[],
            cons=[],
            neutral_points=[],
        )

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = resolve_model_name(os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
    seed_sentiment: dict[str, int | float] = {
        "positive": 0.0,
        "neutral": 0.0,
        "negative": 0.0,
        "total": 0,
        "score": 0.0,
        "confidence": 0.0,
        "positive_count": 0,
        "neutral_count": 0,
        "negative_count": 0,
    }
    fallback_analysis = normalize_analysis_without_fallback(
        build_fallback_analysis(processed_reviews, seed_sentiment),
    )
    logger.info(
        "Preparing AI analysis for %s reviews (%s split reviews sent to Gemini).",
        len(reviews),
        len(processed_reviews),
    )

    if not api_key or api_key in PLACEHOLDER_API_KEYS:
        logger.warning("GEMINI_API_KEY is not configured. Using the local fallback analysis.")
        log_analysis_details(fallback_analysis)
        return fallback_analysis

    try:
        client = genai.Client(api_key=api_key)
        per_review_analyses: list[AIAnalysis] = []

        for review in processed_reviews:
            try:
                per_review_analyses.append(request_review_analysis(client, model_name, review))
            except Exception as exc:
                logger.warning("Single-review Gemini analysis failed, using fallback for one review: %s", exc)
                single_review_fallback = normalize_analysis_without_fallback(
                    build_fallback_analysis([review], seed_sentiment),
                )
                per_review_analyses.append(limit_per_review_analysis_points(single_review_fallback))

        aggregated_analysis = aggregate_review_analyses(per_review_analyses)
        final_summary = generate_final_summary(
            client,
            model_name,
            aggregated_analysis.pros,
            aggregated_analysis.cons,
            aggregated_analysis.summary,
        )
        final_analysis = AIAnalysis(
            summary=final_summary,
            pros=aggregated_analysis.pros,
            cons=aggregated_analysis.cons,
            neutral_points=aggregated_analysis.neutral_points,
        )
        log_analysis_details(final_analysis)
        return final_analysis
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Gemini returned invalid JSON, switching to the local fallback: %s", exc)
        log_analysis_details(fallback_analysis)
        return fallback_analysis
    except Exception as exc:
        logger.warning("Gemini analysis failed, switching to the local fallback: %s", exc)
        log_analysis_details(fallback_analysis)
        return fallback_analysis


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def calculate_percentages(
    positive_weight: float,
    neutral_weight: float,
    negative_weight: float,
    total_weight: float,
) -> tuple[float, float, float]:
    """Distribute weights into percentages using the Largest Remainder Method.

    FIX (type annotation): Parameters were annotated as ``int`` but callers
    pass float sentiment weights.  Renamed from ``*_count`` to ``*_weight``
    and typed as ``float`` to match actual usage.  The Largest Remainder
    Method works correctly with floats; behaviour is unchanged.
    """
    if total_weight == 0:
        return 0.0, 0.0, 0.0

    weights = [positive_weight, neutral_weight, negative_weight]
    raw_percentages = [(w * 10000) / total_weight for w in weights]
    rounded_down = [math.floor(v) for v in raw_percentages]
    remainder = 10000 - sum(rounded_down)
    distribution_order = sorted(
        range(len(weights)),
        key=lambda index: (raw_percentages[index] - rounded_down[index], weights[index]),
        reverse=True,
    )

    for step in range(remainder):
        rounded_down[distribution_order[step % len(distribution_order)]] += 1

    return tuple(round(v / 100, 2) for v in rounded_down)  # type: ignore[return-value]


def calculate_score(positive_weight: float, negative_weight: float, total_weight: float) -> float:
    if total_weight == 0:
        return 0.0

    sentiment_ratio = (positive_weight - negative_weight) / total_weight
    score = (sentiment_ratio + 1) * 2.5
    return round(max(1.0, min(5.0, score)), 2)


def calculate_confidence(
    positive_count: int,
    neutral_count: int,
    negative_count: int,
    total_reviews: int,
) -> float:
    if total_reviews == 0:
        return 0.0

    dominant_count = max(positive_count, neutral_count, negative_count)
    dominant_ratio = dominant_count / total_reviews
    sample_factor = min(math.sqrt(total_reviews / 8), 1.0)
    confidence = dominant_ratio * (0.25 + (0.75 * sample_factor)) * 100
    return round(min(100.0, max(0.0, confidence)), 2)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "AI Product Review Aggregator API is running."}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_reviews(payload: ReviewRequest) -> AnalyzeResponse:
    """Analyse a batch of product reviews and return aggregated insights.

    FIX (Warning): Scope check now runs on the original raw reviews *before*
    ``prepare_reviews`` splits them into fragments.  Previously the check ran
    on post-split fragments, meaning a multi-sentence review with one off-topic
    clause could incorrectly reject an otherwise valid batch.
    """
    try:
        # Scope check on original reviews before any splitting.
        for review in payload.reviews:
            if not is_review_related(review):
                raise HTTPException(status_code=400, detail="Out of scope question is asked.")

        processed_reviews = prepare_reviews(payload.reviews)
        if not processed_reviews:
            raise HTTPException(status_code=400, detail="Please provide at least one non-empty review.")

        logger.info(
            "Received /analyze request with %s raw reviews (%s split reviews processed).",
            len(payload.reviews),
            len(processed_reviews),
        )
        ai_analysis = get_ai_analysis(processed_reviews)
        sentiment = calculate_sentiment_from_reviews(processed_reviews, ai_analysis)
        response_payload = AnalyzeResponse(
            summary=ai_analysis.summary,
            pros=ai_analysis.pros,
            cons=ai_analysis.cons,
            neutral_points=ai_analysis.neutral_points,
            sentiment=SentimentBreakdown(
                positive=sentiment["positive"],
                neutral=sentiment["neutral"],
                negative=sentiment["negative"],
                total=sentiment["total"],
            ),
            score=sentiment["score"],
            confidence=sentiment["confidence"],
        )
        logger.info(
            "Response payload prepared with %s pros, %s cons, score %s, confidence %s, and sentiment %s",
            len(response_payload.pros),
            len(response_payload.cons),
            response_payload.score,
            response_payload.confidence,
            response_payload.sentiment.model_dump(),
        )

        return response_payload
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error while analyzing reviews: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to analyze reviews right now.") from exc