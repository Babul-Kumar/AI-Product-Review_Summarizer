import { useRef, useState } from "react";
import SentimentChart from "./components/SentimentChart";

const ANALYZE_URL = "https://ai-product-review.onrender.com/analyze";
const MIN_REVIEW_INPUT_LENGTH = 10;
const LOW_QUALITY_MESSAGE = "Please enter meaningful product reviews to get accurate insights.";
const NON_MEANINGFUL_PATTERNS = [
  /^no summary was returned\.?$/i,
  /^no major recurring strengths were mentioned\.?$/i,
  /^no major recurring complaints were mentioned\.?$/i,
  /^no neutral feedback found\.?$/i,
];

const EMPTY_RESULT = {
  summary: "",
  pros: [],
  cons: [],
  neutral_points: [],
  sentiment: {
    positive: 0,
    neutral: 0,
    negative: 0,
    total: 0,
  },
  score: 0,
  confidence: 0,
};

function toSafeNumber(value) {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : 0;
}

function toSafeList(value) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.map((item) => String(item).trim()).filter(Boolean);
}

function normalizeResult(data) {
  const sentiment =
    data?.sentiment && typeof data.sentiment === "object" ? data.sentiment : EMPTY_RESULT.sentiment;

  return {
    summary:
      typeof data?.summary === "string" && data.summary.trim()
        ? data.summary.trim()
        : "No summary was returned.",
    pros: toSafeList(data?.pros),
    cons: toSafeList(data?.cons),
    neutral_points: toSafeList(data?.neutral_points),
    sentiment: {
      positive: toSafeNumber(sentiment.positive),
      neutral: toSafeNumber(sentiment.neutral),
      negative: toSafeNumber(sentiment.negative),
      total: toSafeNumber(sentiment.total),
    },
    score: toSafeNumber(data?.score),
    confidence: toSafeNumber(data?.confidence),
  };
}

function formatPercent(value) {
  return `${toSafeNumber(value).toFixed(2)}%`;
}

function formatScore(value) {
  const safeValue = toSafeNumber(value);
  return safeValue > 0 ? `${safeValue.toFixed(2)} / 5` : "Not available";
}

function isMeaningfulItem(item) {
  const text = String(item ?? "").trim();

  return text.length > 0 && !NON_MEANINGFUL_PATTERNS.some((pattern) => pattern.test(text));
}

function hasMeaningfulInsights(data) {
  return data.pros.some(isMeaningfulItem) || data.cons.some(isMeaningfulItem);
}

function App() {
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [lowQualityMessage, setLowQualityMessage] = useState("");

  const inFlightRef = useRef(false);
  const lastRequestKeyRef = useRef("");
  const lastResultRef = useRef(null);

  const reviews = inputText
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const safeResult = result ?? EMPTY_RESULT;
  const trimmedInput = inputText.trim();
  const isInputTooShort = trimmedInput.length > 0 && trimmedInput.length < MIN_REVIEW_INPUT_LENGTH;
  const hasMeaningfulResult = result ? hasMeaningfulInsights(result) : false;
  const isAnalyzeDisabled = loading || reviews.length === 0 || trimmedInput.length < MIN_REVIEW_INPUT_LENGTH;

  const analyzeReviews = async () => {
    if (loading || inFlightRef.current) {
      return;
    }

    const cleanedReviews = inputText
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

    if (cleanedReviews.length === 0) {
      setError("Please enter at least one review before analyzing.");
      setLowQualityMessage("");
      setResult(null);
      return;
    }

    if (trimmedInput.length < MIN_REVIEW_INPUT_LENGTH) {
      setError("");
      setLowQualityMessage(LOW_QUALITY_MESSAGE);
      setResult(null);
      return;
    }

    const requestKey = cleanedReviews.join("\n");

    if (requestKey === lastRequestKeyRef.current && lastResultRef.current) {
      setError("");
      if (hasMeaningfulInsights(lastResultRef.current)) {
        setLowQualityMessage("");
        setResult(lastResultRef.current);
      } else {
        setResult(null);
        setLowQualityMessage(LOW_QUALITY_MESSAGE);
      }
      return;
    }

    setLoading(true);
    setError("");
    setLowQualityMessage("");
    setResult(null);
    inFlightRef.current = true;

    try {
      const response = await fetch(ANALYZE_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ reviews: cleanedReviews }),
      });

      const responseText = await response.text();
      let data = null;

      if (responseText) {
        try {
          data = JSON.parse(responseText);
        } catch {
          throw new Error("The server returned invalid JSON.");
        }
      }

      if (!response.ok) {
        const message =
          typeof data?.detail === "string" ? data.detail : "Unable to analyze reviews right now.";
        throw new Error(message);
      }

      if (!data || typeof data !== "object") {
        throw new Error("The server returned an empty response.");
      }

      const normalizedData = normalizeResult(data);
      lastRequestKeyRef.current = requestKey;
      lastResultRef.current = normalizedData;

      if (hasMeaningfulInsights(normalizedData)) {
        setLowQualityMessage("");
        setResult(normalizedData);
      } else {
        setResult(null);
        setLowQualityMessage(LOW_QUALITY_MESSAGE);
      }
    } catch (requestError) {
      setLowQualityMessage("");
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Something went wrong while connecting to the API.",
      );
    } finally {
      setLoading(false);
      inFlightRef.current = false;
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    await analyzeReviews();
  };

  const renderList = (items, emptyMessage) => {
    const values = items.length > 0 ? items : [emptyMessage];

    return (
      <ul className="mt-4 space-y-3 text-sm leading-6 text-slate-100">
        {values.map((item, index) => (
          <li key={`${item}-${index}`} className="rounded-2xl bg-black/10 px-4 py-3">
            {item}
          </li>
        ))}
      </ul>
    );
  };

  return (
    <main className="relative min-h-screen overflow-hidden px-4 py-10 text-slate-100 sm:px-6 lg:px-8">
      <div aria-hidden="true" className="pointer-events-none absolute inset-0">
        <div className="absolute left-[12%] top-[-4rem] h-72 w-72 rounded-full bg-emerald-300/10 blur-3xl" />
        <div className="absolute right-[-5rem] top-28 h-80 w-80 rounded-full bg-amber-200/8 blur-3xl" />
        <div className="absolute bottom-12 left-1/2 h-64 w-64 -translate-x-1/2 rounded-full bg-teal-300/8 blur-3xl" />
      </div>

      <div className="relative mx-auto flex max-w-6xl flex-col gap-8">
        <section className="overflow-hidden rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.78)] shadow-[0_25px_120px_rgba(4,12,10,0.45)] backdrop-blur-xl">
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_320px] lg:items-center">
            <div className="p-6 sm:p-8 lg:p-10">
              <div className="mb-6 inline-flex rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-emerald-200">
                React Frontend + FastAPI API
              </div>

              <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl">
                AI Product Review Analyzer
              </h1>
              <p className="mt-3 max-w-2xl text-sm text-slate-300/80 sm:text-base">
                Analyze product reviews to extract sentiment, pros, cons, and key insights
                instantly.
              </p>

              <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
                <label className="block">
                  <div className="mb-2 flex flex-wrap items-center justify-between gap-3 text-sm font-medium text-slate-200">
                    <span>Enter Reviews</span>
                    <span className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-300">
                      {reviews.length} review{reviews.length === 1 ? "" : "s"}
                    </span>
                  </div>

                  <textarea
                    value={inputText}
                    onChange={(event) => {
                      setInputText(event.target.value);
                      if (error) {
                        setError("");
                      }
                      if (lowQualityMessage) {
                        setLowQualityMessage("");
                      }
                    }}
                    placeholder={`Battery life is excellent and setup was simple.\nThe camera is average for the price.\nPerformance feels slow sometimes.`}
                    className="min-h-64 w-full rounded-3xl border border-white/10 bg-slate-950/55 px-5 py-4 text-base text-slate-100 outline-none transition focus:border-emerald-300/60 focus:ring-2 focus:ring-emerald-300/20"
                  />
                  <p className="mt-3 text-xs text-slate-400">
                    Enter 2-3 meaningful reviews for better insights
                  </p>
                </label>

                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="text-xs text-slate-500">
                    Live API endpoint configured
                  </div>

                  <button
                    type="submit"
                    disabled={isAnalyzeDisabled}
                    className="inline-flex items-center justify-center rounded-full bg-gradient-to-r from-emerald-400 via-teal-300 to-amber-300 px-6 py-3 text-sm font-semibold text-slate-950 shadow-[0_14px_40px_rgba(52,211,153,0.28)] transition duration-300 hover:-translate-y-0.5 hover:scale-105 disabled:cursor-not-allowed disabled:opacity-70"
                  >
                    {loading ? "Analyzing..." : "Analyze Reviews"}
                  </button>
                </div>
              </form>

              {isInputTooShort && !error && !lowQualityMessage ? (
                <div className="mt-5 rounded-2xl border border-amber-300/20 bg-amber-300/10 px-4 py-3 text-sm text-amber-100">
                  {LOW_QUALITY_MESSAGE}
                </div>
              ) : null}

              {error ? (
                <div className="mt-5 rounded-2xl border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-100">
                  {error}
                </div>
              ) : null}

              {lowQualityMessage ? (
                <div className="mt-5 rounded-2xl border border-amber-300/20 bg-amber-300/10 px-4 py-3 text-sm text-amber-100">
                  {lowQualityMessage}
                </div>
              ) : null}
            </div>

            <div className="px-6 pb-6 sm:px-8 lg:px-0 lg:pr-10">
              <div className="rounded-[1.75rem] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.08),rgba(255,255,255,0.03))] p-5 shadow-[0_18px_50px_rgba(4,12,10,0.22)]">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-amber-100/75">
                  Try Example
                </p>
                <div className="mt-4 rounded-2xl border border-emerald-300/12 bg-black/15 px-4 py-4 text-sm leading-7 text-slate-100">
                  &quot;The camera is great but battery drains fast&quot;
                </div>
              </div>
            </div>
          </div>
        </section>

        {loading ? (
          <section className="rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.72)] p-10 text-center shadow-[0_20px_90px_rgba(4,12,10,0.32)] backdrop-blur-xl">
            <div className="mx-auto h-12 w-12 animate-spin rounded-full border-4 border-emerald-200/20 border-t-emerald-300" />
            <p className="mt-4 text-base text-slate-200">Analyzing...</p>
          </section>
        ) : null}

        {result && hasMeaningfulResult && !isInputTooShort && !lowQualityMessage ? (
          <section className="rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.72)] p-6 shadow-[0_20px_90px_rgba(4,12,10,0.32)] backdrop-blur-xl sm:p-8">
            <div className="mb-8 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-2xl font-semibold text-white">Analysis Result</h2>
                <p className="mt-1 text-sm text-slate-300">
                  Response data from the deployed FastAPI backend.
                </p>
              </div>

              <div className="flex flex-wrap gap-3">
                <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 text-sm">
                  <span className="block text-slate-400">Total Reviews</span>
                  <span className="font-semibold text-slate-100">
                    {safeResult.sentiment.total || reviews.length}
                  </span>
                </div>
                <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 text-sm">
                  <span className="block text-slate-400">Score</span>
                  <span className="font-semibold text-amber-100">{formatScore(safeResult.score)}</span>
                </div>
                <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 text-sm">
                  <span className="block text-slate-400">Confidence</span>
                  <span className="font-semibold text-cyan-200">
                    {formatPercent(safeResult.confidence)}
                  </span>
                </div>
                <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 text-sm">
                  <span className="block text-slate-400">Positive</span>
                  <span className="font-semibold text-emerald-300">
                    {formatPercent(safeResult.sentiment.positive)}
                  </span>
                </div>
                <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 text-sm">
                  <span className="block text-slate-400">Neutral</span>
                  <span className="font-semibold text-amber-200">
                    {formatPercent(safeResult.sentiment.neutral)}
                  </span>
                </div>
                <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 text-sm">
                  <span className="block text-slate-400">Negative</span>
                  <span className="font-semibold text-rose-300">
                    {formatPercent(safeResult.sentiment.negative)}
                  </span>
                </div>
              </div>
            </div>

            <div className="grid gap-6 lg:grid-cols-[1fr_0.95fr]">
              <div className="space-y-6">
                <article className="rounded-[1.5rem] border border-white/10 bg-slate-950/35 p-5">
                  <p className="text-sm uppercase tracking-[0.18em] text-emerald-200/80">Summary</p>
                  <p className="mt-3 text-base leading-7 text-slate-100">{safeResult.summary}</p>
                </article>

                <div className="grid gap-6 md:grid-cols-2">
                  <article className="rounded-[1.5rem] border border-emerald-300/15 bg-emerald-300/8 p-5">
                    <p className="text-sm uppercase tracking-[0.18em] text-emerald-100">Pros</p>
                    {renderList(safeResult.pros, "No pros returned.")}
                  </article>

                  <article className="rounded-[1.5rem] border border-rose-300/15 bg-rose-300/8 p-5">
                    <p className="text-sm uppercase tracking-[0.18em] text-rose-100">Cons</p>
                    {renderList(safeResult.cons, "No cons returned.")}
                  </article>
                </div>

                <article className="rounded-[1.5rem] border border-amber-300/15 bg-amber-300/8 p-5">
                  <p className="text-sm uppercase tracking-[0.18em] text-amber-100">
                    Neutral Points
                  </p>
                  {renderList(safeResult.neutral_points, "No neutral feedback found.")}
                </article>
              </div>

              <div>
                <p className="mb-4 text-sm uppercase tracking-[0.18em] text-amber-100/80">
                  Sentiment Chart
                </p>
                <SentimentChart
                  positive={safeResult.sentiment.positive}
                  neutral={safeResult.sentiment.neutral}
                  negative={safeResult.sentiment.negative}
                  score={safeResult.score}
                  confidence={safeResult.confidence}
                />
              </div>
            </div>
          </section>
        ) : null}

        <footer className="mt-10 text-center text-sm text-slate-400/80">
          Developed by Babul Kumar
        </footer>
      </div>
    </main>
  );
}

export default App;
