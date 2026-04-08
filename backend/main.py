import json
import logging
import math
import os
import re
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
MAX_REVIEWS = 100
MAX_REVIEWS_TO_ANALYZE = 10
SENTIMENT_POLARITY_THRESHOLD = 0.1
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
    "none",
    "not mentioned",
    "n/a",
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


app = FastAPI(title="AI Product Review Aggregator API")

# Allow all origins to support dynamic frontend deployments (e.g., Vercel)
# Fixes CORS preflight (OPTIONS) request failures
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint for monitoring tools (UptimeRobot, etc.)
# Returns simple status to verify server is alive
@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "AI Product Review API",
    }


def is_review_related_rule_based(text: str) -> bool | None:
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


# Hybrid scope detection:
# Rule-based -> fast filtering
# AI fallback -> handles edge cases
def is_review_related(text: str) -> bool:
    rule_based_result = is_review_related_rule_based(text)
    ai_result: bool | None = None

    if rule_based_result is not None:
        logger.info("Scope check: rule=%s, ai=%s", rule_based_result, ai_result)
        return rule_based_result

    try:
        ai_result = is_review_related_ai(text)
        if ai_result:
            logger.info("Scope check: rule=%s, ai=%s", rule_based_result, ai_result)
            return True
    except Exception as exc:
        logger.warning("Hybrid scope detection fallback failed: %s", exc)

    logger.info("Scope check fallback: allowing input (rule=%s, ai=%s)", rule_based_result, ai_result)
    return True


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


def calculate_sentiment(ai_analysis: AIAnalysis) -> dict[str, int | float]:
    positive_count = len(ai_analysis.pros)
    negative_count = len(ai_analysis.cons)
    neutral_count = len(ai_analysis.neutral_points)
    total_points = positive_count + negative_count + neutral_count

    logger.info(
        "Sentiment derived from point counts: %s pros, %s cons, %s neutral points.",
        positive_count,
        negative_count,
        neutral_count,
    )

    if total_points == 0:
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
        positive_count,
        neutral_count,
        negative_count,
        total_points,
    )
    score = calculate_score(positive_count, negative_count, total_points)
    confidence = calculate_confidence(
        positive_count,
        neutral_count,
        negative_count,
        total_points,
    )

    return {
        "positive": positive_percentage,
        "neutral": neutral_percentage,
        "negative": negative_percentage,
        "total": total_points,
        "score": score,
        "confidence": confidence,
        "positive_count": positive_count,
        "neutral_count": neutral_count,
        "negative_count": negative_count,
    }


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


def normalize_point(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" -*\t\r\n")
    cleaned = re.sub(r"^\d+[\).\s-]+", "", cleaned)
    cleaned = normalize_display_text(cleaned)
    if len(cleaned) > 120:
        cleaned = f"{cleaned[:117].rstrip()}..."
    return cleaned


def normalize_display_text(text: str) -> str:
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
    return overlap_count >= 2 and (overlap_count / smaller_set_size) >= 0.6


def should_replace_point(existing: str, candidate: str) -> bool:
    existing_score = (len(point_signature_tokens(existing)), len(existing))
    candidate_score = (len(point_signature_tokens(candidate)), len(candidate))
    return candidate_score > existing_score


def deduplicate_semantic(points: list[str]) -> list[str]:
    result: list[str] = []

    for point in points:
        candidate = normalize_point(point)
        if not candidate:
            continue

        should_skip = False

        for index, existing in enumerate(result):
            if points_overlap(existing, candidate):
                if should_replace_point(existing, candidate):
                    result[index] = candidate
                else:
                    should_skip = True
                break

        if should_skip:
            continue

        if not any(points_overlap(existing, candidate) for existing in result):
            result.append(candidate)

    return result


def clean_point_list(items: list[str], fallback_message: str) -> list[str]:
    items = deduplicate_semantic(list(dict.fromkeys(items)))
    cleaned_points: list[str] = []
    seen: set[str] = set()

    for item in items:
        cleaned_item = normalize_point(item)
        if len(cleaned_item) < 4:
            continue

        point_key = cleaned_item.lower()
        if point_key in seen:
            continue

        seen.add(point_key)
        cleaned_points.append(cleaned_item)

    specific_points = [
        point
        for point in cleaned_points
        if not point.lower().startswith(GENERIC_POINT_MARKERS)
    ]

    if specific_points:
        return specific_points[:MAX_POINTS]

    return cleaned_points[:MAX_POINTS] or [fallback_message]


def clean_optional_point_list(items: list[str]) -> list[str]:
    items = deduplicate_semantic(list(dict.fromkeys(items)))
    cleaned_points: list[str] = []
    seen: set[str] = set()

    for item in items:
        cleaned_item = normalize_point(item)
        if len(cleaned_item) < 4:
            continue

        point_key = cleaned_item.lower()
        if point_key in seen:
            continue

        seen.add(point_key)
        cleaned_points.append(cleaned_item)

    return cleaned_points[:MAX_POINTS]


def filter_generic_neutral_points(points: list[str]) -> list[str]:
    return [
        point
        for point in points
        if "overall" not in point.lower() and "mixed" not in point.lower()
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
    fallback_message: str,
) -> list[str]:
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

    return points or [fallback_message]


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

    pros = clean_point_list(
        extract_points(
            reviews,
            "positive",
            "No major recurring strengths were mentioned.",
        ),
        "No major recurring strengths were mentioned.",
    )
    cons = clean_point_list(
        extract_points(
            reviews,
            "negative",
            "No major recurring complaints were mentioned.",
        ),
        "No major recurring complaints were mentioned.",
    )
    pros = clean_point_list(
        exclude_existing_points(pros, cons),
        "No major recurring strengths were mentioned.",
    )

    neutral_points = clean_optional_point_list(
        extract_points(
            reviews,
            "neutral",
            "",
        )
    )
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

    try:
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
    except Exception as exc:
        logger.warning("AI scope detection failed, falling back to rule-based decision: %s", exc)

    return False


def normalize_analysis(ai_analysis: AIAnalysis) -> AIAnalysis:
    summary = ai_analysis.summary.strip() or "No summary was returned."
    pros = clean_point_list(ai_analysis.pros, "No major recurring strengths were mentioned.")
    cons = clean_point_list(ai_analysis.cons, "No major recurring complaints were mentioned.")
    pros = clean_point_list(
        exclude_existing_points(pros, cons),
        "No major recurring strengths were mentioned.",
    )
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
    pros = exclude_existing_points(pros, cons)
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
    merged_pros = exclude_existing_points(merged_pros, merged_cons)
    merged_neutral_points = exclude_existing_points(merged_neutral_points, merged_pros + merged_cons)

    pros = clean_point_list(
        merged_pros,
        "No major recurring strengths were mentioned.",
    )
    cons = clean_point_list(
        merged_cons,
        "No major recurring complaints were mentioned.",
    )
    neutral_points = clean_optional_point_list(
        merged_neutral_points,
    )
    neutral_points = exclude_existing_points(neutral_points, pros + cons)
    neutral_points = filter_generic_neutral_points(neutral_points)

    return AIAnalysis(
        summary=primary.summary.strip() or secondary.summary,
        pros=pros,
        cons=cons,
        neutral_points=neutral_points,
    )


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
        summary_match = re.search(r"summary\s*:\s*(.+?)(?:\n\s*pros?\s*:|\Z)", cleaned_text, flags=re.I | re.S)
        pros_match = re.search(r"pros?\s*:\s*(.+?)(?:\n\s*cons?\s*:|\Z)", cleaned_text, flags=re.I | re.S)
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


def prepare_reviews(raw_reviews: list[str]) -> list[str]:
    split_reviews: list[str] = []

    for raw_review in raw_reviews:
        for review in raw_review.split("\n"):
            cleaned_review = review.strip()
            if cleaned_review:
                split_reviews.append(cleaned_review)

    if len(split_reviews) > MAX_REVIEWS_TO_ANALYZE:
        logger.info(
            "Received %s split reviews; limiting Gemini analysis to first %s.",
            len(split_reviews),
            MAX_REVIEWS_TO_ANALYZE,
        )

    return split_reviews[:MAX_REVIEWS_TO_ANALYZE]


def deduplicate_points_with_set(points: list[str]) -> list[str]:
    normalized_points: list[str] = []

    for point in points:
        normalized_point = normalize_point(point)
        if not normalized_point:
            continue
        if normalized_point.lower().startswith(GENERIC_POINT_MARKERS):
            continue
        normalized_points.append(normalized_point)

    unique_points = list(set(normalized_points))
    unique_points.sort()
    return deduplicate_semantic(unique_points)


def aggregate_review_analyses(review_analyses: list[AIAnalysis]) -> AIAnalysis:
    all_pros: list[str] = []
    all_cons: list[str] = []
    all_neutral_points: list[str] = []
    summary_parts: list[str] = []

    for analysis in review_analyses:
        all_pros.extend(analysis.pros)
        all_cons.extend(analysis.cons)
        all_neutral_points.extend(analysis.neutral_points)
        if analysis.summary.strip():
            summary_parts.append(analysis.summary.strip())

    pros = deduplicate_points_with_set(all_pros)
    cons = deduplicate_points_with_set(all_cons)
    pros = exclude_existing_points(pros, cons)
    neutral_points = deduplicate_points_with_set(all_neutral_points)
    neutral_points = exclude_existing_points(neutral_points, pros + cons)
    neutral_points = filter_generic_neutral_points(neutral_points)

    fallback_summary = " ".join(summary_parts[:3]).strip()
    if not fallback_summary:
        fallback_summary = "No summary was returned."

    return AIAnalysis(
        summary=fallback_summary,
        pros=pros,
        cons=cons,
        neutral_points=neutral_points,
    )


def build_single_review_prompt(review: str) -> str:
    return f"""
You are an expert product analyst.

Analyze this single customer review.

Tasks:
1. Extract key pros.
2. Extract key cons.
3. Extract neutral or mixed points (optional).
4. Write a one-line summary.

Rules:
* Be specific and feature-based.
* Avoid duplicates.
* Return only details from this review.

Output format (STRICT JSON):
{{
  "summary": "...",
  "pros": ["...", "..."],
  "cons": ["...", "..."],
  "neutral_points": ["...", "..."]
}}

Review:
{review}
""".strip()


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
        return normalize_analysis_without_fallback(validated)

    if raw_response_text:
        parsed_response = parse_ai_response(raw_response_text)
        return normalize_analysis_without_fallback(parsed_response)

    raise ValueError("Gemini returned an empty single-review response.")


def generate_final_summary(
    client: genai.Client,
    model_name: str,
    pros: list[str],
    cons: list[str],
    fallback_summary: str,
) -> str:
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
    seed_sentiment = {
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
                per_review_analyses.append(single_review_fallback)

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


def calculate_percentages(
    positive_count: int,
    neutral_count: int,
    negative_count: int,
    total_reviews: int,
) -> tuple[float, float, float]:
    if total_reviews == 0:
        return 0.0, 0.0, 0.0

    counts = [positive_count, neutral_count, negative_count]
    raw_percentages = [(count * 10000) / total_reviews for count in counts]
    rounded_down = [math.floor(value) for value in raw_percentages]
    remainder = 10000 - sum(rounded_down)
    distribution_order = sorted(
        range(len(counts)),
        key=lambda index: (raw_percentages[index] - rounded_down[index], counts[index]),
        reverse=True,
    )

    for step in range(remainder):
        rounded_down[distribution_order[step % len(distribution_order)]] += 1

    return tuple(round(value / 100, 2) for value in rounded_down)


def calculate_score(positive_count: int, negative_count: int, total_reviews: int) -> float:
    if total_reviews == 0:
        return 0.0

    sentiment_ratio = (positive_count - negative_count) / total_reviews
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


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "AI Product Review Aggregator API is running."}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_reviews(payload: ReviewRequest) -> AnalyzeResponse:
    try:
        processed_reviews = prepare_reviews(payload.reviews)
        if not processed_reviews:
            raise HTTPException(status_code=400, detail="Please provide at least one non-empty review.")

        for review in processed_reviews:
            if not is_review_related(review):
                raise HTTPException(status_code=400, detail="Out of scope question is asked.")

        logger.info(
            "Received /analyze request with %s raw reviews (%s split reviews processed).",
            len(payload.reviews),
            len(processed_reviews),
        )
        ai_analysis = get_ai_analysis(processed_reviews)
        sentiment = calculate_sentiment(ai_analysis)
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
