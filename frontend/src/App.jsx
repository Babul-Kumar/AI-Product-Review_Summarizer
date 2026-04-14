import { useRef, useState, useEffect, useCallback, useMemo } from "react";
import SentimentChart from "./components/SentimentChart";

// ==============================================================================
// CONFIGURATION
// ==============================================================================

const LOG_LEVEL = {
  DEBUG: false,
  INFO: false,
  WARN: true,
  ERROR: true,
};

const log = {
  debug: (...args) => LOG_LEVEL.DEBUG && console.debug("[DEBUG]", ...args),
  info: (...args) => LOG_LEVEL.INFO && console.info("[INFO]", ...args),
  warn: (...args) => LOG_LEVEL.WARN && console.warn("[WARN]", ...args),
  error: (...args) => LOG_LEVEL.ERROR && console.error("[ERROR]", ...args),
};

// Environment Variables
const API_BASE_URL = import.meta.env.VITE_API_URL || "https://ai-product-review.onrender.com";
const ANALYZE_URL = `${API_BASE_URL}/analyze-raw`;
const ANALYZE_STREAM_URL = `${API_BASE_URL}/analyze-stream`;
const PING_URL = `${API_BASE_URL}/ping`;
const API_KEY = import.meta.env.VITE_API_KEY || "";

// Constants
const COOLDOWN_MS = 3000;
const MAX_RETRIES = 2;
const RETRY_DELAY_BASE = 1500;
const REQUEST_TIMEOUT_MS = 20000;
const HEALTH_CHECK_INTERVAL = 30000;
const CACHE_TTL = 120000;
const MAX_INPUT_LENGTH = 5000;
const SLOW_LOADING_THRESHOLD = 5000;

// STREAMING: Stage configurations
const STREAM_STAGES = {
  SUMMARY: { order: 0, name: "summary", icon: "📝", label: "Generating summary..." },
  PROS: { order: 1, name: "pros", icon: "👍", label: "Extracting pros..." },
  CONS: { order: 2, name: "cons", icon: "👎", label: "Identifying cons..." },
  NEUTRAL: { order: 3, name: "neutral_points", icon: "➖", label: "Finding neutral points..." },
  FEATURES: { order: 4, name: "feature_scores", icon: "⭐", label: "Scoring features..." },
  SENTIMENT: { order: 5, name: "sentiment", icon: "📊", label: "Analyzing sentiment..." },
  COMPLETE: { order: 6, name: "complete", icon: "✅", label: "Complete!" },
};

const EMPTY_RESULT = {
  summary: "",
  pros: [],
  cons: [],
  neutral_points: [],
  sentiment: { positive: 0, neutral: 0, negative: 0, total: 0 },
  score: 0,
  confidence: 0,
  feature_scores: [],
};

// ==============================================================================
// GLOBAL ERROR HANDLER
// ==============================================================================

if (typeof window !== "undefined") {
  window.addEventListener("error", (e) => {
    const msg = typeof e.error?.message === "string"
      ? e.error.message
      : e.error?.message || String(e.error?.message || e.message);
    log.error("💥 Global error:", msg);
  });

  window.addEventListener("unhandledrejection", (e) => {
    const reason = e.reason;
    const msg = typeof reason === "string"
      ? reason
      : reason?.message || JSON.stringify(reason);
    log.error("💥 Unhandled promise rejection:", msg);
  });
}

// ==============================================================================
// UTILITY FUNCTIONS
// ==============================================================================

function toSafeNumber(value) {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : 0;
}

function toSafeList(value) {
  if (!Array.isArray(value)) return [];
  
  return value.map((item) => {
    if (typeof item === "string") return item.trim();
    if (item && typeof item === "object") return item.text || "";
    return "";
  }).filter(Boolean);
}

function normalizeResult(data) {
  const sentiment = data?.sentiment && typeof data.sentiment === "object" 
    ? data.sentiment 
    : EMPTY_RESULT.sentiment;

  return {
    summary: typeof data?.summary === "string" && data.summary.trim()
      ? data.summary.trim()
      : "No summary was returned.",
    pros: toSafeList(data?.pros ?? []),
    cons: toSafeList(data?.cons ?? []),
    neutral_points: toSafeList(data?.neutral_points ?? []),
    sentiment: {
      positive: toSafeNumber(sentiment?.positive ?? 0),
      neutral: toSafeNumber(sentiment?.neutral ?? 0),
      negative: toSafeNumber(sentiment?.negative ?? 0),
      total: toSafeNumber(sentiment?.total ?? 0),
    },
    score: toSafeNumber(data?.score ?? 0),
    confidence: toSafeNumber(data?.confidence ?? 0),
    explained_pros: Array.isArray(data?.explained_pros) ? data.explained_pros : [],
    explained_cons: Array.isArray(data?.explained_cons) ? data.explained_cons : [],
    feature_scores: Array.isArray(data?.feature_scores)
      ? data.feature_scores.filter(isValidFeatureScore)
      : [],
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
  return text.length > 0;
}

function hasMeaningfulInsights(data) {
  if (!data) return false;
  const hasPros = Array.isArray(data.pros) && data.pros.some(isMeaningfulItem);
  const hasCons = Array.isArray(data.cons) && data.cons.some(isMeaningfulItem);
  return hasPros || hasCons;
}

function extractFeatureScore(featureScore) {
  if (featureScore && typeof featureScore === "object" && !Array.isArray(featureScore)) {
    return {
      feature: String(featureScore.feature || "").trim(),
      score: toSafeNumber(featureScore.score),
    };
  }
  
  if (Array.isArray(featureScore) && featureScore.length >= 2) {
    return {
      feature: String(featureScore[0] || "").trim(),
      score: toSafeNumber(featureScore[1]),
    };
  }
  
  if (typeof featureScore === "string" && featureScore.trim()) {
    return {
      feature: featureScore.trim(),
      score: 0,
    };
  }
  
  return { feature: "", score: 0 };
}

function isValidFeatureScore(featureScore) {
  const { feature, score } = extractFeatureScore(featureScore);
  if (!feature && score === 0) return false;
  if (feature === "Unknown feature" || feature === "") return false;
  return true;
}

function createPartialResult() {
  return {
    summary: null,
    pros: null,
    cons: null,
    neutral_points: null,
    feature_scores: null,
    sentiment: null,
    score: null,
    confidence: null,
    _meta: {
      completedStages: [],
      streaming: true,
    }
  };
}

function updatePartialResult(current, type, data) {
  const newResult = { 
    ...current,
    _meta: current._meta ? { ...current._meta } : { completedStages: [], streaming: true }
  };
  
  const alreadyTracked = newResult._meta.completedStages.includes(type);
  
  switch (type) {
    case "summary":
      newResult.summary = data;
      break;
    case "pros":
      newResult.pros = toSafeList(data ?? []);
      break;
    case "cons":
      newResult.cons = toSafeList(data ?? []);
      break;
    case "neutral_points":
      newResult.neutral_points = toSafeList(data ?? []);
      break;
    case "feature_scores":
      newResult.feature_scores = Array.isArray(data) ? data : [];
      break;
    case "sentiment":
      newResult.sentiment = data;
      newResult.score = toSafeNumber(data?.score ?? 0);
      newResult.confidence = toSafeNumber(data?.confidence ?? 0);
      break;
    case "complete":
      break;
  }
  
  if (!alreadyTracked) {
    newResult._meta.completedStages.push(type);
  }
  
  return newResult;
}

// ==============================================================================
// FETCH UTILITIES
// ==============================================================================

// ✅ FIX 3: Use options.signal OR controller.signal
function fetchWithTimeout(url, options, timeout = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort();
  }, timeout);

  // ✅ FIX 3: Preserve incoming signal if provided
  return fetch(url, {
    ...options,
    signal: options.signal || controller.signal,
  }).finally(() => clearTimeout(timeoutId));
}

async function fetchWithRetry(url, options, retries = MAX_RETRIES, retryCount = 0, requestId = "unknown") {
  try {
    const response = await fetchWithTimeout(url, options);

    if (response.ok) {
      log.debug(`✅ [${requestId}] Success on attempt ${retryCount + 1}`);
      return { success: true, response, retriesUsed: retryCount };
    }

    const shouldRetry = response.status >= 500 || 
                        response.status === 429 || 
                        response.status === 503 || 
                        response.status === 504;

    if (shouldRetry && retryCount < retries) {
      const delay = RETRY_DELAY_BASE * Math.pow(2, retryCount);
      log.debug(`🔄 [${requestId}] Retry ${retryCount + 1}/${retries} after ${delay}ms`);
      await new Promise((resolve) => setTimeout(resolve, delay));
      return fetchWithRetry(url, options, retries, retryCount + 1, requestId);
    }

    return { success: false, response, retriesUsed: retryCount };
  } catch (error) {
    const isAbort = error.name === "AbortError" || error.message?.includes("aborted");
    const isTimeout = isAbort || error.message?.includes("timeout");

    log.debug(`❌ [${requestId}] Attempt ${retryCount + 1} failed: ${error.message}`);

    if (retryCount < retries && !isAbort) {
      const delay = RETRY_DELAY_BASE * Math.pow(2, retryCount);
      await new Promise((resolve) => setTimeout(resolve, delay));
      return fetchWithRetry(url, options, retries, retryCount + 1, requestId);
    }

    const finalError = new Error(isTimeout 
      ? `Request timed out after ${(retryCount + 1) * REQUEST_TIMEOUT_MS}ms` 
      : error.message);
    finalError.isTimeout = isTimeout;
    finalError.isAbort = isAbort;
    throw finalError;
  }
}

async function fetchWithStreaming(url, options, onChunk, onStage, signal, requestId, onFallback) {
  const startTime = Date.now();
  
  try {
    if (signal?.aborted) {
      throw { isAbort: true, message: "Request cancelled" };
    }
    
    const response = await fetchWithTimeout(url, {
      ...options,
      signal,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    if (!reader) {
      throw new Error("Streaming not supported");
    }

    while (true) {
      if (signal?.aborted) {
        reader.cancel();
        throw { isAbort: true, message: "Request cancelled" };
      }
      
      const { done, value } = await reader.read();
      
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      
      const lines = buffer.split("\n");
      buffer = lines.length > 0 ? lines.pop() : "";
      
      for (const line of lines) {
        const clean = line.trim();
        if (!clean || clean.toUpperCase() === "[DONE]") continue;
        
        let parsed;
        try {
          // ✅ FIX 4: Parse clean instead of line
          parsed = JSON.parse(clean);
        } catch (e) {
          log.debug(`[Stream] Skipped malformed JSON: ${clean.substring(0, 50)}...`);
          continue;
        }
        
        const { type, data } = parsed;
        
        log.debug(`[Stream:${type}]`, data);
        onChunk(type, data);
        
        const stage = STREAM_STAGES[type.toUpperCase()] || STREAM_STAGES[type];
        if (stage) {
          onStage(stage, data);
        }
      }
    }

    const duration = Date.now() - startTime;
    return { success: true, duration };

  } catch (error) {
    if (error.isAbort || error.name === "AbortError") {
      throw { isAbort: true, message: "Request cancelled" };
    }
    
    if (onFallback) {
      onFallback(error.message);
    }
    throw error;
  }
}

// ==============================================================================
// HELPER: Safe message extraction
// ==============================================================================

function extractMessage(data) {
  const rawMessage = data?.detail || data?.message || null;
  
  if (!rawMessage) {
    return { message: "Unknown error", errorCode: null };
  }
  
  if (typeof rawMessage === "string") {
    return { message: rawMessage, errorCode: null };
  }
  
  if (typeof rawMessage === "object") {
    return {
      message: rawMessage?.message || rawMessage?.msg || JSON.stringify(rawMessage),
      errorCode: rawMessage?.error || rawMessage?.code || rawMessage?.type || null
    };
  }
  
  return { message: String(rawMessage), errorCode: null };
}

// ==============================================================================
// MAIN COMPONENT
// ==============================================================================

function App() {
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [serverStatus, setServerStatus] = useState("checking");
  const [lastClickTime, setLastClickTime] = useState(0);
  const [copySuccess, setCopySuccess] = useState(false);
  const [retryInfo, setRetryInfo] = useState(null);
  const [loadingStage, setLoadingStage] = useState("");
  const [usingCachedResult, setUsingCachedResult] = useState(false);
  const [isSlowLoading, setIsSlowLoading] = useState(false);
  const [analysisStats, setAnalysisStats] = useState(null);
  const [showStats, setShowStats] = useState(false);
  const [streamingFallback, setStreamingFallback] = useState(false);
  const streamingFallbackTimerRef = useRef(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStage, setCurrentStage] = useState(null);
  const [stageProgress, setStageProgress] = useState({});

  const requestIdRef = useRef(null);
  const abortControllerRef = useRef(null);
  const inFlightRef = useRef(false);
  const lastRequestKeyRef = useRef("");
  const lastResultRef = useRef(null);
  const lastCacheTimeRef = useRef(0);
  const isMountedRef = useRef(true);

  // ==============================================================================
  // EFFECTS
  // ==============================================================================

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (streamingFallbackTimerRef.current) {
        clearTimeout(streamingFallbackTimerRef.current);
      }
    };
  }, []);

  // ✅ FIX 10: Add timeout to health check
  useEffect(() => {
    const checkServerHealth = async () => {
      try {
        const res = await fetchWithTimeout(PING_URL, { method: "GET" }, 5000);

        if (isMountedRef.current) {
          setServerStatus(res.ok ? "online" : "offline");
        }
      }
      catch {
        if (isMountedRef.current) {
          setServerStatus("offline");
        }
      }
    };

    checkServerHealth();
    const intervalId = setInterval(checkServerHealth, HEALTH_CHECK_INTERVAL);
    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    if (!loading) {
      setLoadingStage("");
      setIsSlowLoading(false);
      setCurrentStage(null);
      return;
    }

    const steps = [
      "📥 Reading reviews...",
      "🧠 Understanding context...",
      "📊 Analyzing patterns...",
      "✨ Preparing results..."
    ];

    let i = 0;
    setLoadingStage(steps[i]);
    setIsSlowLoading(false);

    const stageInterval = setInterval(() => {
      i = (i + 1) % steps.length;
      if (isMountedRef.current) {
        setLoadingStage(steps[i]);
      }
    }, 1500);

    const slowTimer = setTimeout(() => {
      if (loading && isMountedRef.current && !isStreaming) {
        setIsSlowLoading(true);
      }
    }, SLOW_LOADING_THRESHOLD);

    return () => {
      clearInterval(stageInterval);
      clearTimeout(slowTimer);
    };
  }, [loading, isStreaming]);

  // ✅ FIX 1: Proper cleanup for both timeouts
  useEffect(() => {
    if (result) {
      const timer = setTimeout(() => {
        document.getElementById("result-section")?.scrollIntoView({
          behavior: "smooth",
          block: "start"
        });

        setShowStats(true);

        // ✅ FIX 1: Inner timeout with cleanup
        const hideTimer = setTimeout(() => {
          if (isMountedRef.current) {
            setShowStats(false);
          }
        }, 3000);

        // Return cleanup for inner timeout
        return () => clearTimeout(hideTimer);
      }, 100);
      
      // Cleanup for outer timeout
      return () => clearTimeout(timer);
    }
  }, [result]);

  // ==============================================================================
  // HELPERS
  // ==============================================================================

  const reviews = useMemo(() => {
    return inputText
      .split(/\n+/)
      .map(line => line.trim())
      .filter(line => line.length > 0);
  }, [inputText]);
  
  const safeResult = result ?? EMPTY_RESULT;
  const trimmedInput = inputText.trim();
  const hasMeaningfulResult = result ? hasMeaningfulInsights(result) : false;
  const isInCooldown = Date.now() - lastClickTime < COOLDOWN_MS;
  
  const hasPartialContent = result && (
    (result.summary?.length ?? 0) > 0 ||
    (result.pros?.length ?? 0) > 0 ||
    (result.cons?.length ?? 0) > 0
  );

  const hasInput = trimmedInput.length > 0;
  
  const isAnalyzeDisabled = 
    loading ||
    !hasInput ||
    isInCooldown ||
    serverStatus === "offline";

  // ==============================================================================
  // ACTIONS
  // ==============================================================================

  const copyToClipboard = () => {
    if (!navigator.clipboard) {
      log.warn("[Copy] Clipboard API not available");
      setError("Copy not supported in this browser");
      return;
    }

    // ✅ FIX 9: Safe copy with fallback value
    const textToCopy = JSON.stringify({
      summary: safeResult.summary,
      pros: safeResult.pros,
      cons: safeResult.cons,
      neutral_points: safeResult.neutral_points,
      sentiment: safeResult.sentiment,
      score: safeResult.score,
      confidence: safeResult.confidence,
      feature_scores: safeResult.feature_scores,
      analysisTime: analysisStats?.duration || 0,
    }, null, 2);

    navigator.clipboard.writeText(textToCopy).then(() => {
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    }).catch((err) => {
      log.error("[Copy] Failed:", err);
      setError("Failed to copy to clipboard");
    });
  };

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      log.info("[Cancel] Request cancelled by user");
    }
  };

  const handleInputChange = (e) => {
    const value = e.target.value;
    
    if (value.length > MAX_INPUT_LENGTH) {
      setInputText(value.slice(0, MAX_INPUT_LENGTH));
      setError(`Input truncated to ${MAX_INPUT_LENGTH} characters.`);
      return;
    }
    
    setInputText(value);
    setError("");
  };

  const handleStreamChunk = useCallback((type, data) => {
    setResult(prev => {
      const base = prev ?? createPartialResult();
      return updatePartialResult({ ...base }, type, data);
    });
    
    setStageProgress(prev => ({
      ...prev,
      [type]: "complete"
    }));
  }, []);

  const handleStreamStage = useCallback((stage, data) => {
    setCurrentStage(stage);
    setStageProgress(prev => ({
      ...prev,
      [stage.name]: "loading"
    }));
  }, []);

  const handleStreamingFallback = useCallback((errorMessage) => {
    setStreamingFallback(true);
    log.warn("[Streaming] Falling back to regular request:", errorMessage);
    
    if (streamingFallbackTimerRef.current) {
      clearTimeout(streamingFallbackTimerRef.current);
    }
    
    streamingFallbackTimerRef.current = setTimeout(() => {
      if (isMountedRef.current) {
        setStreamingFallback(false);
      }
    }, 5000);
  }, []);

  const analyzeReviews = async () => {
    if (inFlightRef.current) return;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const newRequestId = `REQ-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    requestIdRef.current = newRequestId;

    const rawReviews = reviews;

    if (rawReviews.length === 0) {
      setError("Please enter at least one review or piece of text to analyze.");
      return;
    }

    const now = Date.now();
    if (now - lastClickTime < COOLDOWN_MS) {
      setError("Too many requests. Please wait a few seconds.");
      return;
    }
    setLastClickTime(now);

    const requestKey = JSON.stringify(rawReviews.map(r => r.trim().toLowerCase()));
    const cacheValid = 
      requestKey === lastRequestKeyRef.current &&
      lastResultRef.current &&
      Date.now() - lastCacheTimeRef.current < CACHE_TTL;

    if (cacheValid) {
      setError("");
      setUsingCachedResult(true);
      
      if (hasMeaningfulInsights(lastResultRef.current)) {
        setResult(lastResultRef.current);
      }
      setRetryInfo(null);
      return;
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setLoading(true);
    setError("");
    setUsingCachedResult(false);
    setRetryInfo({ retries: 0, maxRetries: MAX_RETRIES, status: "connecting" });
    setShowStats(false);
    setCurrentStage(null);
    setStageProgress({});
    setStreamingFallback(false);
    inFlightRef.current = true;

    const startTime = Date.now();
    
    const analysisTimerLabel = `analysis-${newRequestId}`;
    console.time(analysisTimerLabel);
    log.info(`[Analytics] Analysis started: ${newRequestId}`);

    const headers = { "Content-Type": "application/json" };
    if (API_KEY) headers["X-API-Key"] = API_KEY;

    const fetchOptions = {
      method: "POST",
      headers,
      body: JSON.stringify({ 
        raw_text: inputText.trim()
      }),
    };

    try {
      setIsStreaming(true);
      
      const streamResult = await fetchWithStreaming(
        ANALYZE_STREAM_URL,
        fetchOptions,
        handleStreamChunk,
        handleStreamStage,
        controller.signal,
        newRequestId,
        handleStreamingFallback
      );
      
      if (streamResult.success) {
        const duration = Date.now() - startTime;
        console.timeEnd(analysisTimerLabel);
        
        const stats = {
          duration,
          streaming: true,
          reviewsAnalyzed: rawReviews.length,
        };
        
        setAnalysisStats(stats);
        
        // ✅ FIX 6: Check for sentiment.total specifically
        setResult(prev => {
          if (!prev || !prev.sentiment || prev.sentiment.total === undefined) return prev;
          const normalizedResult = normalizeResult(prev);
          lastRequestKeyRef.current = requestKey;
          lastResultRef.current = normalizedResult;
          lastCacheTimeRef.current = Date.now();
          return normalizedResult;
        });
        
        log.info(`[Analytics] ✅ Streaming completed:`, stats);
        
        setRetryInfo(null);
        
        if (isMountedRef.current) {
          setIsStreaming(false);
          setLoading(false);
        }
        return;
      }
    } catch (streamError) {
      if (!streamError.isAbort) {
        handleStreamingFallback(streamError.message);
        // ✅ FIX 2: Abort controller to prevent double request
        controller.abort();
      }
      log.warn("[Streaming] Fallback to regular request:", streamError.message);
    }

    // ✅ FIX 5: DON'T set streaming false here - let fallback finish naturally
    // REMOVED: setIsStreaming(false);
    
    const { response, retriesUsed } = await fetchWithRetry(
      ANALYZE_URL, fetchOptions, MAX_RETRIES, 0, newRequestId
    );

    setRetryInfo((prev) => prev ? { ...prev, retries: retriesUsed, status: retriesUsed > 0 ? "retried" : "success" } : null);

    if (controller.signal.aborted) {
      setError("Request was cancelled. Please try again.");
      setLoading(false);
      inFlightRef.current = false;
      console.timeEnd(analysisTimerLabel);
      return;
    }

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
      const { message, errorCode } = extractMessage(data);
      let userMessage = message;
      let errorType = "generic";

      if (response.status === 422) {
        userMessage = "⚠️ Invalid input format. Please try different text.";
        errorType = "validation";
      }
      else if (errorCode === "INVALID_API_KEY") {
        userMessage = "🔑 Backend API keys not configured. Please try again later.";
        errorType = "config";
      }
      else if (errorCode === "OUT_OF_SCOPE") {
        userMessage = "❌ Input is not a valid product review. Please provide text to analyze.";
        errorType = "validation";
      }
      else if (errorCode === "AI_PROCESSING_FAILED" || errorCode === "GEMINI_ERROR") {
        userMessage = "🤖 AI processing failed. Server may be busy — please try again.";
        errorType = "ai";
      }
      else if (errorCode === "QUEUE_FULL" || errorCode === "RATE_LIMITED") {
        userMessage = "⚠️ Server queue is full. Please wait and try again.";
        errorType = "queue";
      }
      else if (response.status === 429) {
        userMessage = "⚠️ API rate limit reached. Please wait a few seconds.";
        errorType = "rate_limit";
      }
      else if (response.status >= 500) {
        userMessage = "🛠️ Server error. The team has been notified.";
        errorType = "server";
      }

      log.error(`[Error:${errorType}] ${message} (code: ${errorCode})`);
      throw new Error(userMessage);
    }

    if (!data || typeof data !== "object") {
      throw new Error("The server returned an empty response.");
    }

    const normalizedData = normalizeResult(data);
    
    lastRequestKeyRef.current = requestKey;
    lastResultRef.current = normalizedData;
    lastCacheTimeRef.current = Date.now();

    const endTime = Date.now();
    const duration = endTime - startTime;
    console.timeEnd(analysisTimerLabel);
    
    const stats = {
      duration,
      retriesUsed,
      reviewsAnalyzed: rawReviews.length,
      responseSize: responseText.length,
      cached: false,
      streaming: false,
    };
    
    setAnalysisStats(stats);
    log.info(`[Analytics] ✅ Analysis completed:`, stats);

    if (hasMeaningfulInsights(normalizedData)) {
      setResult(normalizedData);
    } else {
      setError("The analysis didn't return meaningful results. Try with different text.");
    }
    setRetryInfo(null);

    } catch (requestError) {
      if (controller.signal.aborted || requestError.isAbort) {
        setError("Request was cancelled. Please try again.");
        setLoading(false);
        inFlightRef.current = false;
        console.timeEnd(analysisTimerLabel);
        return;
      }

      let errorMessage;
      let errorType = "network";
      const errorText = requestError.message || String(requestError);

      if (requestError.isTimeout || errorText.includes("timed out") || errorText.includes("timeout")) {
        errorMessage = "⏱️ AI is taking longer than usual. Please try again.";
        errorType = "timeout";
      } else if (
        errorText.includes("Failed to fetch") ||
        errorText.includes("NetworkError") ||
        errorText.includes("fetch failed") ||
        errorText.includes("network")
      ) {
        errorMessage = "🔌 Network issue detected. Please check your connection.";
        errorType = "network";
      } else if (errorText.includes("429")) {
        errorMessage = "⚠️ API rate limit reached. Please wait a few seconds.";
        errorType = "rate_limit";
      } else if (errorText.includes("CORS") || errorText.includes("cors")) {
        errorMessage = "🌐 CORS error. Please try again from a different browser.";
        errorType = "cors";
      } else {
        errorMessage = errorText || "Something went wrong. Please try again.";
        errorType = "unknown";
      }

      log.error(`[Error:${errorType}] ${errorText}`);

      if (lastResultRef.current && hasMeaningfulInsights(lastResultRef.current)) {
        log.warn("[Fallback] Using cached result due to error:", errorType);
        setError("");
        setUsingCachedResult(true);
        setResult(lastResultRef.current);
        setRetryInfo(null);
      } else {
        setError(errorMessage);
        setRetryInfo(null);
      }
    } finally {
      if (isMountedRef.current) {
        setLoading(false);
        setIsStreaming(false);
        setCurrentStage(null);
        inFlightRef.current = false;
        // ✅ FIX 7: Use null for consistency
        abortControllerRef.current = null;
      }
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    analyzeReviews();
  };

  // ==============================================================================
  // RENDER HELPERS
  // ==============================================================================

  const renderList = (items, emptyMessage) => {
    const values = items && items.length > 0 ? items : [emptyMessage];
    return (
      <ul className="mt-4 space-y-3 text-sm leading-6 text-slate-100">
        {values.map((item, index) => {
          const text = typeof item === "string" ? item : item.text;
          const feature = typeof item === "object" && item?.feature ? item.feature : null;
          
          return (
            <li key={`item-${index}`} className="rounded-2xl bg-black/10 px-4 py-3">
              {text}
              {feature && (
                <span className="ml-2 text-xs text-slate-400">
                  ({feature})
                </span>
              )}
            </li>
          );
        })}
      </ul>
    );
  };

  const renderFeatureScore = (feature, score, maxScore = 5) => {
    const safeScore = toSafeNumber(score);
    const percentage = Math.min((safeScore / maxScore) * 100, 100);
    
    const getScoreColor = (s) => {
      if (s >= 4) return "bg-emerald-400";
      if (s >= 3) return "bg-amber-400";
      return "bg-rose-400";
    };

    return (
      <div className="mt-3">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-slate-200 capitalize">
            {(feature || "").replace(/_/g, " ") || "Unknown feature"}
          </span>
          <span className="text-slate-300 font-medium">
            {safeScore.toFixed(1)}/{maxScore}
          </span>
        </div>
        <div className="h-2 rounded-full bg-black/30 overflow-hidden">
          <div 
            className={`h-full rounded-full transition-all duration-500 ${getScoreColor(safeScore)}`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    );
  };

  const renderEmptyChart = () => (
    <div className="flex h-64 flex-col items-center justify-center rounded-2xl border border-white/10 bg-slate-950/20">
      <div className="text-4xl mb-3">📊</div>
      <p className="text-sm text-slate-400">No sentiment data available</p>
      <p className="text-xs text-slate-500 mt-1">Submit text to see the chart</p>
    </div>
  );

  const renderSkeleton = () => (
    <section className="rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.72)] p-6 sm:p-8 animate-pulse">
      <div className="mb-8">
        <div className="h-8 w-48 rounded-lg bg-white/5" />
        <div className="mt-2 h-4 w-64 rounded bg-white/5" />
      </div>
      <div className="mb-6 grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
        {[...Array(6)].map((_, i) => <div key={i} className="h-20 rounded-2xl bg-white/5" />)}
      </div>
      <div className="grid gap-6 lg:grid-cols-[1fr_0.95fr]">
        <div className="space-y-6">
          <div className="h-32 rounded-[1.5rem] bg-white/5" />
          <div className="grid gap-6 md:grid-cols-2">
            <div className="h-40 rounded-[1.5rem] bg-white/5" />
            <div className="h-40 rounded-[1.5rem] bg-white/5" />
          </div>
        </div>
        <div className="h-64 rounded-2xl bg-white/5" />
      </div>
    </section>
  );

  const renderStreamingProgress = () => (
    <section className="rounded-[2rem] border border-emerald-400/20 bg-[rgba(7,23,19,0.72)] p-6 sm:p-8">
      <div className="mb-6">
        <h2 className="text-2xl font-semibold text-white flex items-center gap-3">
          <span className="animate-pulse">🔄</span>
          Streaming Results
        </h2>
        <p className="mt-1 text-sm text-emerald-300/80">
          Processing in real-time...
        </p>
        
        {streamingFallback && (
          <div className="mt-2 text-xs text-amber-300 animate-pulse">
            ⚠️ Streaming unavailable, using standard mode...
          </div>
        )}
      </div>

      <div className="space-y-3 mb-8">
        {Object.values(STREAM_STAGES).filter(s => s.name !== "complete").map((stage) => {
          const status = stageProgress[stage.name] || "pending";
          const isActive = currentStage?.name === stage.name;
          
          return (
            <div 
              key={stage.name}
              className={`flex items-center gap-3 p-3 rounded-xl transition-all ${
                isActive ? "bg-emerald-400/10 border border-emerald-400/30" : "bg-white/5"
              }`}
            >
              <span className="text-xl">
                {status === "complete" ? "✅" : isActive ? stage.icon : "⏳"}
              </span>
              <div className="flex-1">
                <p className={`text-sm font-medium ${
                  status === "complete" ? "text-emerald-300" : isActive ? "text-white" : "text-slate-400"
                }`}>
                  {stage.label}
                </p>
                {isActive && (
                  <div className="mt-1 h-1 rounded-full bg-emerald-400/30 overflow-hidden">
                    <div className="h-full w-1/2 bg-emerald-400 animate-pulse rounded-full" />
                  </div>
                )}
              </div>
              <span className={`text-xs ${
                status === "complete" ? "text-emerald-400" : isActive ? "text-amber-300" : "text-slate-500"
              }`}>
                {status === "complete" ? "Done" : isActive ? "Loading..." : "Waiting"}
              </span>
            </div>
          );
        })}
      </div>

      {hasPartialContent && (
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
            Partial Results (more coming...)
          </h3>
          
          {result?.summary && result.summary.length > 0 && (
            <div className="rounded-xl border border-white/10 bg-slate-950/35 p-4">
              <p className="text-xs uppercase tracking-wider text-emerald-200/80 mb-2">Summary</p>
              <p className="text-sm text-slate-100">{result.summary}</p>
            </div>
          )}
          
          {result?.pros?.length > 0 && (
            <div className="rounded-xl border border-emerald-300/15 bg-emerald-300/8 p-4">
              <p className="text-xs uppercase tracking-wider text-emerald-100 mb-2">Pros (preliminary)</p>
              {renderList(result.pros.slice(0, 2), "Loading...")}
            </div>
          )}
          
          {result?.cons?.length > 0 && (
            <div className="rounded-xl border border-rose-300/15 bg-rose-300/8 p-4">
              <p className="text-xs uppercase tracking-wider text-rose-100 mb-2">Cons (preliminary)</p>
              {renderList(result.cons.slice(0, 2), "Loading...")}
            </div>
          )}
        </div>
      )}
    </section>
  );

  // ==============================================================================
  // RENDER
  // ==============================================================================

  return (
    <main className="relative min-h-screen overflow-hidden px-4 py-10 text-slate-100 sm:px-6 lg:px-8">
      <div aria-hidden="true" className="pointer-events-none absolute inset-0">
        <div className="absolute left-[12%] top-[-4rem] h-72 w-72 rounded-full bg-emerald-300/10 blur-3xl" />
        <div className="absolute right-[-5rem] top-28 h-80 w-80 rounded-full bg-amber-200/8 blur-3xl" />
        <div className="absolute bottom-12 left-1/2 h-64 w-64 -translate-x-1/2 rounded-full bg-teal-300/8 blur-3xl" />
      </div>

      <div className="relative mx-auto flex max-w-6xl flex-col gap-8">
        {/* Header */}
        <section className="overflow-hidden rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.78)]">
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_320px] lg:items-center">
            <div className="p-6 sm:p-8 lg:p-10">
              <div className="mb-6 flex flex-wrap items-center gap-3">
                <div className="inline-flex rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-emerald-200">
                  React + FastAPI
                </div>
                <div className={`flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ${
                  serverStatus === "online"
                    ? "border border-emerald-400/30 bg-emerald-400/10 text-emerald-300"
                    : serverStatus === "offline"
                    ? "border border-rose-400/30 bg-rose-400/10 text-rose-300"
                    : "border border-slate-400/30 bg-slate-400/10 text-slate-400"
                }`}>
                  <span className={`h-2 w-2 rounded-full ${
                    serverStatus === "online" ? "bg-emerald-400 animate-pulse" :
                    serverStatus === "offline" ? "bg-rose-400" : "animate-pulse bg-slate-400"
                  }`} />
                  {serverStatus === "online" ? "Server Online" :
                   serverStatus === "offline" ? "Server Offline" : "Checking..."}
                </div>
              </div>

              <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-white">
                AI Product Review Analyzer
              </h1>
              <p className="mt-3 max-w-2xl text-sm text-slate-300/80 sm:text-base">
                Analyze product reviews to extract sentiment, pros, cons, and key insights instantly.
              </p>

              <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
                <label className="block">
                  <div className="mb-2 flex flex-wrap items-center justify-between gap-3 text-sm font-medium text-slate-200">
                    <span>Enter Reviews or Text</span>
                    <span className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-300">
                      {reviews.length} entry{reviews.length === 1 ? "" : "s"}
                    </span>
                  </div>

                  <div className="relative">
                    <textarea
                      value={inputText}
                      onChange={handleInputChange}
                      placeholder={`Battery life is excellent and setup was simple.\nThe camera is average for the price.\nPerformance feels slow sometimes.\n\nSeparate reviews with newlines!`}
                      className="min-h-48 sm:min-h-64 w-full rounded-3xl border border-white/10 bg-slate-950/55 px-4 sm:px-5 py-3 sm:py-4 text-base text-slate-100 outline-none transition focus:border-emerald-300/60 focus:ring-2 focus:ring-emerald-300/20 resize-y"
                      disabled={loading}
                      rows={6}
                    />
                  </div>
                  
                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
                    <span>
                      Separate reviews with newlines. AI will analyze everything.
                    </span>
                    <span className="text-slate-500">
                      {inputText.length}/{MAX_INPUT_LENGTH}
                    </span>
                  </div>
                </label>

                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex flex-col gap-1">
                    <div className="text-xs text-slate-500 hidden sm:block">
                      API: {API_BASE_URL.replace(/^https?:\/\//, "")}
                    </div>
                    {isInCooldown && !error && (
                      <div className="text-xs text-amber-300 animate-pulse">⏳ Cooldown active...</div>
                    )}
                    {serverStatus === "offline" && !error && (
                      <div className="text-xs text-rose-300">🔴 Server is offline. Please wait or refresh.</div>
                    )}
                    {retryInfo?.status === "retried" && !error && (
                      <div className="text-xs text-emerald-400 animate-pulse">
                        ✅ Recovered after {retryInfo.retries} retry{retryInfo.retries !== 1 ? "s" : ""}
                      </div>
                    )}
                    {isSlowLoading && (
                      <div className="text-xs text-amber-300 animate-pulse">
                        ⏱️ Taking longer than usual...
                      </div>
                    )}
                    {showStats && analysisStats && (
                      <div className="text-xs text-cyan-300 animate-pulse">
                        ⚡ Completed in {analysisStats.duration}ms
                      </div>
                    )}
                    {usingCachedResult && (
                      <div className="text-xs text-amber-400">📦 Using cached result</div>
                    )}
                  </div>

                  <div className="flex flex-col items-start sm:items-end gap-2">
                    <button
                      type="submit"
                      disabled={isAnalyzeDisabled}
                      className="w-full sm:w-auto inline-flex items-center justify-center rounded-full bg-gradient-to-r from-emerald-400 via-teal-300 to-amber-300 px-6 py-3 text-sm font-semibold text-slate-950 shadow-[0_14px_40px_rgba(52,211,153,0.28)] transition duration-300 hover:-translate-y-0.5 hover:scale-105 disabled:cursor-not-allowed disabled:opacity-70"
                    >
                      {loading ? (
                        <span className="flex items-center gap-2">
                          <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                          </svg>
                          {isStreaming ? (
                            <span className="flex items-center gap-2">
                              🔄 Streaming... 
                              {currentStage?.icon} {currentStage?.label}
                            </span>
                          ) : (
                            loadingStage || "Analyzing..."
                          )}
                        </span>
                      ) : (
                        "Analyze Reviews"
                      )}
                    </button>
                    
                    {loading && (
                      <button
                        type="button"
                        onClick={handleCancel}
                        className="text-xs text-rose-400 hover:text-rose-300 hover:underline transition-colors"
                      >
                        Cancel request
                      </button>
                    )}
                  </div>
                </div>
              </form>

              {error && (
                <div className="mt-5 rounded-2xl border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-100">
                  {error}
                </div>
              )}
            </div>

            <div className="px-6 pb-6 sm:px-8 lg:px-0 lg:pr-10">
              <div className="rounded-[1.75rem] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.08),rgba(255,255,255,0.03))] p-5">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-amber-100/75">Try Example</p>
                <div 
                  className="mt-4 rounded-2xl border border-emerald-300/12 bg-black/15 px-4 py-4 text-sm leading-7 text-slate-100 cursor-pointer hover:bg-black/25 transition"
                  onClick={() => setInputText("The camera is great but battery drains fast. Performance is excellent for the price. Display is bright but could be sharper. Sound quality is average.")}
                >
                  "The camera is great but battery drains fast. Performance is excellent for the price."
                </div>
              </div>
            </div>
          </div>
        </section>

        {loading && isStreaming && renderStreamingProgress()}
        {loading && !isStreaming && renderSkeleton()}

        {!result && !loading && (
          <div className="text-center text-slate-400 py-12 sm:py-16 rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.72)]">
            <div className="text-4xl sm:text-5xl mb-4">🧠</div>
            <p className="text-base sm:text-lg text-slate-200">Start analyzing product reviews</p>
            <p className="text-sm mt-3 text-slate-500">
              Paste multiple reviews separated by new lines
            </p>
          </div>
        )}

        {result && hasMeaningfulResult && !loading && (
          <section id="result-section" className="rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.72)] p-4 sm:p-6 lg:p-8">
            <div className="mb-6 sm:mb-8 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-xl sm:text-2xl font-semibold text-white">Analysis Result</h2>
                <p className="mt-1 text-sm text-slate-300">
                  AI-powered analysis complete.
                  {usingCachedResult && (
                    <span className="ml-2 text-xs text-amber-400">(cached result)</span>
                  )}
                </p>
                
                {analysisStats && (
                  <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-slate-500">
                    <span className="flex items-center gap-1">
                      ⚡ {analysisStats.duration}ms
                      {analysisStats.streaming && <span className="text-emerald-400">[Streaming]</span>}
                    </span>
                    <span>•</span>
                    <span>{analysisStats.reviewsAnalyzed} entries</span>
                    {analysisStats.retriesUsed > 0 && (
                      <>
                        <span>•</span>
                        <span className="text-amber-400">{analysisStats.retriesUsed} retries</span>
                      </>
                    )}
                  </div>
                )}
              </div>
              <button
                onClick={copyToClipboard}
                className={`w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition ${
                  copySuccess ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-300" : "border-white/10 bg-white/5 text-slate-300 hover:border-white/20"
                }`}
              >
                {copySuccess ? (
                  <>
                    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    Copied!
                  </>
                ) : (
                  <>
                    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                    Copy Results
                  </>
                )}
              </button>
            </div>

            <div className="mb-6 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 sm:gap-3">
              <div className="rounded-xl sm:rounded-2xl border border-white/10 bg-black/10 px-3 sm:px-4 py-2 sm:py-3 text-xs sm:text-sm">
                <span className="block text-slate-400 text-xs">Entries</span>
                <span className="font-semibold text-slate-100">{safeResult.sentiment.total || reviews.length}</span>
              </div>
              <div className="rounded-xl sm:rounded-2xl border border-white/10 bg-black/10 px-3 sm:px-4 py-2 sm:py-3 text-xs sm:text-sm">
                <span className="block text-slate-400 text-xs">Score</span>
                <span className="font-semibold text-amber-100">{formatScore(safeResult.score)}</span>
              </div>
              <div className="rounded-xl sm:rounded-2xl border border-white/10 bg-black/10 px-3 sm:px-4 py-2 sm:py-3 text-xs sm:text-sm">
                <span className="block text-slate-400 text-xs">Confidence</span>
                <span className="font-semibold text-cyan-200">{formatPercent(safeResult.confidence)}</span>
              </div>
              <div className="rounded-xl sm:rounded-2xl border border-white/10 bg-black/10 px-3 sm:px-4 py-2 sm:py-3 text-xs sm:text-sm">
                <span className="block text-slate-400 text-xs">Positive</span>
                <span className="font-semibold text-emerald-300">{formatPercent(safeResult.sentiment.positive)}</span>
              </div>
              <div className="rounded-xl sm:rounded-2xl border border-white/10 bg-black/10 px-3 sm:px-4 py-2 sm:py-3 text-xs sm:text-sm">
                <span className="block text-slate-400 text-xs">Neutral</span>
                <span className="font-semibold text-amber-200">{formatPercent(safeResult.sentiment.neutral)}</span>
              </div>
              <div className="rounded-xl sm:rounded-2xl border border-white/10 bg-black/10 px-3 sm:px-4 py-2 sm:py-3 text-xs sm:text-sm">
                <span className="block text-slate-400 text-xs">Negative</span>
                <span className="font-semibold text-rose-300">{formatPercent(safeResult.sentiment.negative)}</span>
              </div>
            </div>

            <div className="grid gap-6 lg:grid-cols-[1fr_0.95fr]">
              <div className="space-y-4 sm:space-y-6">
                <article className="rounded-xl sm:rounded-[1.5rem] border border-white/10 bg-slate-950/35 p-4 sm:p-5">
                  <p className="text-xs sm:text-sm uppercase tracking-[0.18em] text-emerald-200/80">Summary</p>
                  <p className="mt-2 sm:mt-3 text-sm sm:text-base leading-6 sm:leading-7 text-slate-100">{safeResult.summary}</p>
                </article>

                <div className="grid gap-4 sm:gap-6 md:grid-cols-2">
                  <article className="rounded-xl sm:rounded-[1.5rem] border border-emerald-300/15 bg-emerald-300/8 p-4 sm:p-5">
                    <p className="text-xs sm:text-sm uppercase tracking-[0.18em] text-emerald-100">Pros</p>
                    {renderList(safeResult.pros, "No pros returned.")}
                  </article>
                  <article className="rounded-xl sm:rounded-[1.5rem] border border-rose-300/15 bg-rose-300/8 p-4 sm:p-5">
                    <p className="text-xs sm:text-sm uppercase tracking-[0.18em] text-rose-100">Cons</p>
                    {renderList(safeResult.cons, "No cons returned.")}
                  </article>
                </div>

                <article className="rounded-xl sm:rounded-[1.5rem] border border-amber-300/15 bg-amber-300/8 p-4 sm:p-5">
                  <p className="text-xs sm:text-sm uppercase tracking-[0.18em] text-amber-100">Neutral Points</p>
                  {renderList(safeResult.neutral_points, "No neutral feedback found.")}
                </article>

                {/* ✅ FIX 8: Removed duplicate filter - already filtered in normalizeResult */}
                {safeResult.feature_scores?.length > 0 && (
                  <article className="rounded-xl sm:rounded-[1.5rem] border border-cyan-300/15 bg-cyan-300/8 p-4 sm:p-5">
                    <p className="text-xs sm:text-sm uppercase tracking-[0.18em] text-cyan-100">Feature Scores</p>
                    <div className="space-y-1">
                      {safeResult.feature_scores.map((featureScore, index) => {
                        const { feature, score } = extractFeatureScore(featureScore);
                        return (
                          <div key={`feature-${index}`}>
                            {renderFeatureScore(feature, score)}
                          </div>
                        );
                      })}
                    </div>
                  </article>
                )}
              </div>

              <div className="mt-6 lg:mt-0">
                <p className="mb-4 text-xs sm:text-sm uppercase tracking-[0.18em] text-amber-100/80">Sentiment Chart</p>
                <div className="w-full overflow-hidden rounded-2xl border border-white/10 bg-slate-950/20">
                  {safeResult.sentiment.total > 0 ? (
                    <div className="w-full">
                      <SentimentChart
                        positive={safeResult.sentiment.positive}
                        neutral={safeResult.sentiment.neutral}
                        negative={safeResult.sentiment.negative}
                        score={safeResult.score}
                        confidence={safeResult.confidence}
                      />
                    </div>
                  ) : (
                    renderEmptyChart()
                  )}
                </div>
              </div>
            </div>
          </section>
        )}

        <footer className="mt-10 text-center text-xs sm:text-sm text-slate-400/80">
          Developed by Babul Kumar • Ultra-resilient with auto-retry + timeout
        </footer>
      </div>
    </main>
  );
}

export default App;
