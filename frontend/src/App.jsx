import { useRef, useState } from "react";
import SentimentChart from "./components/SentimentChart";

const ANALYZE_URL = "https://ai-product-review.onrender.com/analyze";

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

function App() {
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const inFlightRef = useRef(false);
  const lastRequestKeyRef = useRef("");
  const lastResultRef = useRef(null);

  const reviews = inputText
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const safeResult = result ?? EMPTY_RESULT;

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
      setResult(null);
      return;
    }

    const requestKey = cleanedReviews.join("\n");

    if (requestKey === lastRequestKeyRef.current && lastResultRef.current) {
      setError("");
      setResult(lastResultRef.current);
      return;
    }

    setLoading(true);
    setError("");
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
      setResult(normalizedData);
    } catch (requestError) {
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
    <main className="min-h-screen px-4 py-10 text-slate-100 sm:px-6 lg:px-8">
      <div className="mx-auto flex max-w-6xl flex-col gap-8">
        <section className="overflow-hidden rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.78)] shadow-[0_25px_120px_rgba(4,12,10,0.45)] backdrop-blur-xl">
          <div className="grid gap-8 lg:grid-cols-[1.05fr_0.95fr]">
            <div className="p-6 sm:p-8 lg:p-10">
              <div className="mb-6 inline-flex rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-emerald-200">
                React Frontend + FastAPI API
              </div>

              <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl">
                AI Product Review Analyzer
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300 sm:text-lg">
                Paste one review per line, send them to the deployed FastAPI backend, and view the
                summary, pros, cons, neutral points, sentiment breakdown, score, and confidence.
              </p>

              <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
                <label className="block">
                  <div className="mb-2 flex flex-wrap items-center justify-between gap-3 text-sm font-medium text-slate-200">
                    <span>Product Reviews</span>
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
                    }}
                    placeholder={`Battery life is excellent and setup was simple.\nThe camera is average for the price.\nPerformance feels slow sometimes.`}
                    className="min-h-64 w-full rounded-3xl border border-white/10 bg-slate-950/55 px-5 py-4 text-base text-slate-100 outline-none transition focus:border-emerald-300/60 focus:ring-2 focus:ring-emerald-300/20"
                  />
                  <p className="mt-3 text-sm text-slate-400">
                    Each line becomes one review in `inputText.split("\n")`.
                  </p>
                </label>

                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="rounded-2xl border border-emerald-300/15 bg-emerald-300/8 px-4 py-3 text-sm text-emerald-100">
                    API: {ANALYZE_URL}
                  </div>

                  <button
                    type="submit"
                    disabled={loading || reviews.length === 0}
                    className="inline-flex items-center justify-center rounded-full bg-gradient-to-r from-emerald-400 via-teal-300 to-amber-300 px-6 py-3 text-sm font-semibold text-slate-950 shadow-[0_14px_40px_rgba(52,211,153,0.28)] transition duration-300 hover:-translate-y-0.5 hover:scale-105 disabled:cursor-not-allowed disabled:opacity-70"
                  >
                    {loading ? "Analyzing..." : "Analyze Reviews"}
                  </button>
                </div>
              </form>

              {error ? (
                <div className="mt-5 rounded-2xl border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-100">
                  {error}
                </div>
              ) : null}
            </div>

            <div className="border-l-0 border-white/10 bg-[linear-gradient(180deg,rgba(15,36,31,0.95),rgba(9,18,17,0.8))] p-6 sm:p-8 lg:border-l lg:p-10">
              <div className="rounded-[1.75rem] border border-white/10 bg-white/5 p-5">
                <p className="text-sm uppercase tracking-[0.22em] text-amber-200/80">
                  What this shows
                </p>
                <ul className="mt-5 space-y-4 text-sm text-slate-200">
                  <li className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3">
                    Summary generated from all submitted reviews.
                  </li>
                  <li className="rounded-2xl border border-emerald-300/15 bg-emerald-300/8 px-4 py-3">
                    Pros highlighted in a green card.
                  </li>
                  <li className="rounded-2xl border border-rose-300/15 bg-rose-300/8 px-4 py-3">
                    Cons highlighted in a red card.
                  </li>
                  <li className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3">
                    Sentiment percentages, score, confidence, and a live pie chart.
                  </li>
                </ul>
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

        {result ? (
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
