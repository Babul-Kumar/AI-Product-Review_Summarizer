import { useEffect, useRef, useState } from "react";
import SentimentChart from "./components/SentimentChart";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const ANALYZE_URL = `${API_BASE_URL}/analyze`;
const MIN_RECOMMENDED_REVIEWS = 2;
const MAX_REVIEW_CHARACTERS = 5000;
const EMPTY_ANALYSIS = {
  summary: "",
  pros: [],
  cons: [],
  neutral_points: [],
  score: null,
  confidence: 0,
  sentiment: {
    positive: 0,
    neutral: 0,
    negative: 0,
    total: 0,
  },
};

function normalizeAnalysis(payload) {
  return {
    summary: typeof payload?.summary === "string" ? payload.summary : "",
    pros: Array.isArray(payload?.pros) ? payload.pros : [],
    cons: Array.isArray(payload?.cons) ? payload.cons : [],
    neutral_points: Array.isArray(payload?.neutral_points) ? payload.neutral_points : [],
    score: Number.isFinite(Number(payload?.score)) ? Number(payload.score) : null,
    confidence: Number(payload?.confidence) || 0,
    sentiment: {
      positive: Number(payload?.sentiment?.positive) || 0,
      neutral: Number(payload?.sentiment?.neutral) || 0,
      negative: Number(payload?.sentiment?.negative) || 0,
      total: Number(payload?.sentiment?.total) || 0,
    },
  };
}

async function readErrorMessage(response) {
  const fallbackMessage = "The backend could not analyze the reviews.";

  try {
    const payload = await response.json();
    return typeof payload?.detail === "string" ? payload.detail : fallbackMessage;
  } catch {
    return fallbackMessage;
  }
}

function App() {
  const [reviewText, setReviewText] = useState("");
  const [reviewCount, setReviewCount] = useState(0);
  const [analysis, setAnalysis] = useState(EMPTY_ANALYSIS);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [copyStatus, setCopyStatus] = useState("");
  const [resultsVisible, setResultsVisible] = useState(false);
  const resultsRef = useRef(null);
  const copyTimeoutRef = useRef(null);
  const lastRequestRef = useRef({ key: "", result: null });

  const hasOnlyWhitespaceInput = reviewText.length > 0 && reviewText.trim().length === 0;
  const hasTooFewReviews = reviewText.trim().length > 0 && reviewCount > 0 && reviewCount < MIN_RECOMMENDED_REVIEWS;
  const isCharacterLimitExceeded = reviewText.length > MAX_REVIEW_CHARACTERS;
  const showInlineValidation = hasOnlyWhitespaceInput || hasTooFewReviews;

  useEffect(() => {
    const count = reviewText
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean).length;

    setReviewCount(count);
  }, [reviewText]);

  useEffect(() => {
    if (!analysis.summary) {
      setResultsVisible(false);
      return undefined;
    }

    setResultsVisible(false);
    const frameId = window.requestAnimationFrame(() => {
      setResultsVisible(true);
      if (resultsRef.current) {
        resultsRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });

    return () => {
      window.cancelAnimationFrame(frameId);
    };
  }, [analysis.summary]);

  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) {
        window.clearTimeout(copyTimeoutRef.current);
      }
    };
  }, []);

  const handleInputChange = (event) => {
    setReviewText(event.target.value);
    if (error) {
      setError("");
    }
  };

  const handleClear = () => {
    setReviewText("");
    setAnalysis(EMPTY_ANALYSIS);
    setError("");
    setCopyStatus("");

    if (copyTimeoutRef.current) {
      window.clearTimeout(copyTimeoutRef.current);
      copyTimeoutRef.current = null;
    }
  };

  const handleCopySummary = async () => {
    if (!analysis.summary) {
      return;
    }

    try {
      if (!navigator?.clipboard?.writeText) {
        throw new Error("Clipboard is unavailable");
      }

      await navigator.clipboard.writeText(analysis.summary);
      setCopyStatus("Copied");
    } catch {
      setCopyStatus("Copy failed");
    }

    if (copyTimeoutRef.current) {
      window.clearTimeout(copyTimeoutRef.current);
    }

    copyTimeoutRef.current = window.setTimeout(() => {
      setCopyStatus("");
      copyTimeoutRef.current = null;
    }, 2000);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    const reviews = reviewText
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

    if (reviews.length === 0) {
      setError("Please enter at least one review before analyzing.");
      setAnalysis(EMPTY_ANALYSIS);
      return;
    }

    if (isCharacterLimitExceeded) {
      setError(`Please shorten your reviews to ${MAX_REVIEW_CHARACTERS} characters or fewer.`);
      return;
    }

    const requestKey = reviews.join("\n");
    if (lastRequestRef.current.key === requestKey && lastRequestRef.current.result) {
      setError("");
      setCopyStatus("");
      setAnalysis(lastRequestRef.current.result);
      return;
    }

    setLoading(true);
    setError("");
    setAnalysis(EMPTY_ANALYSIS);
    setCopyStatus("");

    try {
      const response = await fetch(ANALYZE_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ reviews }),
      });

      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }

      const data = await response.json();
      const normalizedData = normalizeAnalysis(data);
      lastRequestRef.current = {
        key: requestKey,
        result: normalizedData,
      };
      setAnalysis(normalizedData);
    } catch (requestError) {
      setError(
        requestError.message ||
          "Unable to connect to the backend. Make sure FastAPI is running on port 8000.",
      );
    } finally {
      setLoading(false);
    }
  };

  const formatPercentage = (value) => `${Number(value || 0).toFixed(1)}%`;
  const formatScore = (value) =>
    typeof value === "number" && Number.isFinite(value) ? value.toFixed(1) : "--";

  return (
    <main className="min-h-screen px-4 py-10 text-slate-100 sm:px-6 lg:px-8">
      <div className="mx-auto flex max-w-6xl flex-col gap-8">
        <section className="overflow-hidden rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.78)] shadow-[0_25px_120px_rgba(4,12,10,0.45)] backdrop-blur-xl">
          <div className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
            <div className="p-6 sm:p-8 lg:p-10">
              <div className="mb-8 flex items-center gap-3">
                <span className="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.25em] text-emerald-200">
                  Hybrid NLP
                </span>
                <span className="text-sm text-slate-300">React + FastAPI + Gemini + TextBlob</span>
              </div>

              <h1 className="max-w-2xl text-4xl font-bold tracking-tight text-white sm:text-5xl">
                AI Product Review Aggregator
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300 sm:text-lg">
                Paste one review per line to generate a concise summary, extract pros and cons,
                and visualize the overall sentiment.
              </p>

              <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
                <label className="block">
                  <div className="mb-2 flex items-center justify-between text-sm font-medium text-slate-200">
                    <span>Product Reviews</span>
                    <div className="flex items-center gap-2">
                      <span className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-300">
                        {reviewCount} review{reviewCount === 1 ? "" : "s"}
                      </span>
                      <span className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-300">
                        {reviewText.length}/{MAX_REVIEW_CHARACTERS}
                      </span>
                    </div>
                  </div>
                  <textarea
                    value={reviewText}
                    onChange={handleInputChange}
                    placeholder={`The battery lasts all day and the screen looks great.\nShipping was fast, but the camera struggles in low light.\nExcellent value for the price and very easy to set up.`}
                    className="min-h-64 w-full rounded-3xl border border-white/10 bg-slate-950/55 px-5 py-4 text-base text-slate-100 outline-none transition focus:border-emerald-300/60 focus:ring-2 focus:ring-emerald-300/20"
                  />
                  {showInlineValidation ? (
                    <p className="mt-3 text-sm text-amber-200">
                      Please enter valid reviews (one per line)
                    </p>
                  ) : null}
                  {isCharacterLimitExceeded ? (
                    <p className="mt-3 text-sm text-rose-200">
                      Review input is too long. Keep it within {MAX_REVIEW_CHARACTERS} characters.
                    </p>
                  ) : null}
                </label>

                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <p className="text-sm text-slate-400">
                    Tip: each new line becomes one review in the backend request.
                  </p>
                  <div className="flex flex-col gap-3 sm:flex-row">
                    <button
                      type="button"
                      onClick={handleClear}
                      disabled={loading}
                      className="inline-flex items-center justify-center rounded-full border border-white/10 bg-white/5 px-5 py-3 text-sm font-medium text-slate-200 transition hover:border-white/20 hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      Clear
                    </button>
                    <button
                      type="submit"
                      disabled={loading || reviewCount === 0 || isCharacterLimitExceeded}
                      aria-busy={loading}
                      className="inline-flex items-center justify-center rounded-full bg-gradient-to-r from-emerald-400 via-teal-300 to-amber-300 px-6 py-3 text-sm font-semibold text-slate-950 shadow-[0_14px_40px_rgba(52,211,153,0.28)] transition hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-70"
                    >
                      {loading ? "Analyzing..." : "Analyze Reviews"}
                    </button>
                  </div>
                </div>
              </form>

              {error ? (
                <div className="mt-5 flex items-start gap-3 rounded-2xl border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-100 shadow-[0_16px_40px_rgba(251,113,133,0.08)]">
                  <span aria-hidden="true" className="text-base leading-5">
                    {"\u26A0\uFE0F"}
                  </span>
                  <div>
                    <p className="font-medium text-rose-50">We couldn&apos;t analyze the reviews just yet.</p>
                    <p className="mt-1 text-rose-100/90">{error}</p>
                  </div>
                </div>
              ) : null}
            </div>

            <div className="border-l-0 border-white/10 bg-[linear-gradient(180deg,rgba(15,36,31,0.95),rgba(9,18,17,0.8))] p-6 sm:p-8 lg:border-l lg:p-10">
              <div className="rounded-[1.75rem] border border-white/10 bg-white/5 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]">
                <p className="text-sm uppercase tracking-[0.22em] text-amber-200/80">
                  What you get
                </p>
                <ul className="mt-5 space-y-4 text-sm text-slate-200">
                  <li className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3">
                    Local TextBlob sentiment percentages for fast scoring.
                  </li>
                  <li className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3">
                    Gemini-powered summary, pros, and cons extraction.
                  </li>
                  <li className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3">
                    A clean sentiment chart that updates with every analysis.
                  </li>
                </ul>
              </div>

              <div className="mt-6 rounded-[1.75rem] border border-emerald-300/15 bg-emerald-300/8 p-5">
                <p className="text-sm font-medium text-emerald-100">Backend endpoint</p>
                <p className="mt-2 font-mono text-sm text-emerald-200">{ANALYZE_URL}</p>
              </div>
            </div>
          </div>
        </section>

        {loading ? (
          <section className="rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.72)] p-10 text-center shadow-[0_20px_90px_rgba(4,12,10,0.32)] backdrop-blur-xl">
            <div className="mx-auto h-12 w-12 animate-spin rounded-full border-4 border-emerald-200/20 border-t-emerald-300" />
            <p className="mt-4 text-base text-slate-200">Generating insights from your reviews...</p>
          </section>
        ) : null}

        {analysis.summary ? (
          <section
            ref={resultsRef}
            className="rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.72)] p-6 shadow-[0_20px_90px_rgba(4,12,10,0.32)] backdrop-blur-xl sm:p-8"
            style={{
              opacity: resultsVisible ? 1 : 0,
              transform: resultsVisible ? "translateY(0)" : "translateY(14px)",
              transition: "opacity 320ms ease, transform 320ms ease",
            }}
          >
            <div className="mb-8 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-2xl font-semibold text-white">Analysis Result</h2>
                <p className="mt-1 text-sm text-slate-300">
                  Summary, key themes, and sentiment breakdown from the submitted reviews.
                </p>
              </div>
              <div className="flex flex-wrap gap-3">
                <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 text-sm">
                  <span className="block text-slate-400">Score</span>
                  <span className="font-semibold text-amber-100">{formatScore(analysis.score)} ⭐</span>
                </div>
                <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 text-sm">
                  <span className="block text-slate-400">Confidence</span>
                  <span className="font-semibold text-cyan-200">
                    {formatPercentage(analysis.confidence)}
                  </span>
                </div>
                <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 text-sm">
                  <span className="block text-slate-400">Positive</span>
                  <span className="font-semibold text-emerald-300">
                    {formatPercentage(analysis.sentiment.positive)}
                  </span>
                </div>
                <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 text-sm">
                  <span className="block text-slate-400">Neutral</span>
                  <span className="font-semibold text-amber-200">
                    {formatPercentage(analysis.sentiment.neutral)}
                  </span>
                </div>
                <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 text-sm">
                  <span className="block text-slate-400">Negative</span>
                  <span className="font-semibold text-rose-300">
                    {formatPercentage(analysis.sentiment.negative)}
                  </span>
                </div>
              </div>
            </div>

            <div className="grid gap-6 lg:grid-cols-[1fr_0.95fr]">
              <div className="space-y-6">
                <article className="rounded-[1.5rem] border border-white/10 bg-slate-950/35 p-5">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-sm uppercase tracking-[0.18em] text-emerald-200/80">Summary</p>
                    <button
                      type="button"
                      onClick={handleCopySummary}
                      className="inline-flex items-center justify-center rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-white/20 hover:bg-white/10"
                    >
                      {copyStatus || "Copy"}
                    </button>
                  </div>
                  <p className="mt-3 text-base leading-7 text-slate-100">{analysis.summary}</p>
                </article>

                <div className="grid gap-6 md:grid-cols-2">
                  <article className="rounded-[1.5rem] border border-emerald-300/15 bg-emerald-300/8 p-5">
                    <p className="text-sm uppercase tracking-[0.18em] text-emerald-100">Pros</p>
                    <ul className="mt-4 space-y-3 text-sm leading-6 text-slate-100">
                      {analysis.pros.map((item, index) => (
                        <li key={`pro-${index}`} className="rounded-2xl bg-black/10 px-3 py-2">
                          {item}
                        </li>
                      ))}
                    </ul>
                  </article>

                  <article className="rounded-[1.5rem] border border-rose-300/15 bg-rose-300/8 p-5">
                    <p className="text-sm uppercase tracking-[0.18em] text-rose-100">Cons</p>
                    <ul className="mt-4 space-y-3 text-sm leading-6 text-slate-100">
                      {analysis.cons.map((item, index) => (
                        <li key={`con-${index}`} className="rounded-2xl bg-black/10 px-3 py-2">
                          {item}
                        </li>
                      ))}
                    </ul>
                  </article>
                </div>

                {analysis.neutral_points.length ? (
                  <article className="rounded-[1.5rem] border border-amber-300/15 bg-amber-300/8 p-5">
                    <p className="text-sm uppercase tracking-[0.18em] text-amber-100">
                      Neutral Points
                    </p>
                    <ul className="mt-4 space-y-3 text-sm leading-6 text-slate-100">
                      {analysis.neutral_points.map((item, index) => (
                        <li key={`neutral-${index}`} className="rounded-2xl bg-black/10 px-3 py-2">
                          {item}
                        </li>
                      ))}
                    </ul>
                  </article>
                ) : null}
              </div>

              <div>
                <p className="mb-4 text-sm uppercase tracking-[0.18em] text-amber-100/80">
                  Sentiment Chart
                </p>
                <SentimentChart
                  positive={Number(analysis.sentiment.positive) || 0}
                  neutral={Number(analysis.sentiment.neutral) || 0}
                  negative={Number(analysis.sentiment.negative) || 0}
                  score={analysis.score}
                  confidence={analysis.confidence}
                />
                <p className="mt-4 text-sm leading-6 text-slate-300">
                  Confidence reflects review volume and how consistently the sentiment leans in one
                  direction.
                </p>
              </div>
            </div>
          </section>
        ) : null}
      </div>
    </main>
  );
}

export default App;
