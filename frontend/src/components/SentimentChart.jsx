import { useEffect, useMemo, useRef, useState } from "react";
import Chart from "chart.js/auto";

// Modified: register the center text plugin once for the app lifecycle.
const centerTextPlugin = {
  id: "sentimentCenterText",
  beforeDraw(chart, args, pluginOptions) {
    const { ctx, chartArea } = chart;
    const text = pluginOptions?.text || "Sentiment";

    if (!chartArea) {
      return;
    }

    const centerX = (chartArea.left + chartArea.right) / 2;
    const centerY = (chartArea.top + chartArea.bottom) / 2;
    const chartWidth = chartArea.right - chartArea.left;
    const maxTextWidth = chartWidth * 0.54;
    let fontSize = Math.max(12, Math.min(chartWidth / 10, 22));

    ctx.save();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#f8fafc";
    ctx.font = `600 ${fontSize}px Space Grotesk`;

    const measuredTextWidth = ctx.measureText(text).width;
    if (measuredTextWidth > maxTextWidth) {
      fontSize = Math.max(11, Math.floor(fontSize * (maxTextWidth / measuredTextWidth)));
      ctx.font = `600 ${fontSize}px Space Grotesk`;
    }

    ctx.fillText(text, centerX, centerY);
    ctx.restore();
  },
};

if (!globalThis.__sentimentCenterTextPluginRegistered) {
  Chart.register(centerTextPlugin);
  globalThis.__sentimentCenterTextPluginRegistered = true;
}

function SentimentChart({ positive, neutral, negative, score, confidence }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const lastLoggedConfidenceRef = useRef(null);
  const previousTotalRef = useRef(0);

  // Modified: sanitize incoming values and memoize chart inputs.
  const chartValues = useMemo(
    () =>
      [positive, neutral, negative].map((value) =>
        typeof value === "number" && Number.isFinite(value) ? value : 0,
      ),
    [negative, neutral, positive],
  );
  const [safePositive, safeNeutral, safeNegative] = chartValues;
  const total = useMemo(
    () => chartValues.reduce((sum, value) => sum + value, 0),
    [chartValues],
  );
  const [isChartVisible, setIsChartVisible] = useState(total > 0);

  // Modified: keep center text safe and stable across renders.
  const centerText = useMemo(
    () =>
      total === 0
        ? "No Data"
        : typeof score === "number" && Number.isFinite(score)
        ? `\u2B50 ${score.toFixed(1)}`
        : "Sentiment",
    [score, total],
  );
  const ariaLabel = useMemo(
    () =>
      `Sentiment chart showing ${safePositive.toFixed(1)}% positive, ` +
      `${safeNeutral.toFixed(1)}% neutral, and ${safeNegative.toFixed(1)}% negative`,
    [safeNegative, safeNeutral, safePositive],
  );

  // Modified: fade the chart in when data becomes available again.
  useEffect(() => {
    if (total === 0) {
      setIsChartVisible(false);
      previousTotalRef.current = 0;
      return undefined;
    }

    let frameId;
    if (previousTotalRef.current === 0) {
      setIsChartVisible(false);
      frameId = requestAnimationFrame(() => {
        setIsChartVisible(true);
      });
    } else {
      setIsChartVisible(true);
    }

    previousTotalRef.current = total;

    return () => {
      if (frameId) {
        cancelAnimationFrame(frameId);
      }
    };
  }, [total]);

  useEffect(() => {
    if (!canvasRef.current) {
      return undefined;
    }

    if (total === 0 && chartRef.current) {
      chartRef.current.canvas.style.cursor = "default";
      chartRef.current.destroy();
      chartRef.current = null;
      return undefined;
    }

    if (total === 0) {
      canvasRef.current.style.cursor = "default";
      return undefined;
    }

    if (!chartRef.current) {
      chartRef.current = new Chart(canvasRef.current, {
        type: "pie",
        data: {
          labels: ["Positive", "Neutral", "Negative"],
          datasets: [
            {
              label: "Sentiment",
              data: [safePositive, safeNeutral, safeNegative],
              backgroundColor: [
                "rgba(34,197,94,0.85)",
                "rgba(234,179,8,0.85)",
                "rgba(244,63,94,0.85)",
              ],
              // Modified: brighter hover colors for a clearer highlight effect.
              hoverBackgroundColor: [
                "rgba(74,222,128,0.98)",
                "rgba(250,204,21,0.98)",
                "rgba(251,113,133,0.98)",
              ],
              borderColor: ["#0f172a", "#0f172a", "#0f172a"],
              borderWidth: 2,
              // Modified: subtle visual polish for hover and segment separation.
              spacing: 2,
              hoverBorderColor: "rgba(248,250,252,0.9)",
              hoverBorderWidth: 3,
              hoverOffset: 14,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: "58%",
          // Modified: hover cursor feedback for interactive slices.
          onHover(event, elements, chart) {
            const canvas = chart?.canvas || event?.native?.target;

            if (canvas) {
              canvas.style.cursor = elements.length > 0 ? "pointer" : "default";
            }
          },
          animation: {
            animateRotate: true,
            duration: 1000,
          },
          plugins: {
            centerText: {
              text: centerText,
            },
            legend: {
              position: "bottom",
              labels: {
                color: "#e2e8f0",
                padding: 18,
                font: {
                  family: "Space Grotesk",
                  size: 13,
                },
                generateLabels(chart) {
                  const labels = chart.data.labels || [];
                  const dataset = chart.data.datasets[0];

                  return labels.map((label, index) => ({
                    text: `${label}: ${Number(dataset.data[index] || 0).toFixed(1)}%`,
                    fillStyle: dataset.backgroundColor[index],
                    strokeStyle: dataset.borderColor[index],
                    lineWidth: dataset.borderWidth,
                    hidden: !chart.getDataVisibility(index),
                    index,
                  }));
                },
              },
            },
            tooltip: {
              callbacks: {
                // Modified: cleaner spacing and safer numeric formatting.
                label(context) {
                  return ` ${context.label}: ${Number(context.parsed ?? 0).toFixed(1)}% (out of total sentiment)`;
                },
              },
            },
          },
        },
      });
      return undefined;
    }

    // Modified: update only the changing pieces to avoid unnecessary chart work.
    const nextData = [safePositive, safeNeutral, safeNegative];
    const dataset = chartRef.current.data.datasets[0];
    const hasDataChanged = dataset.data.some((value, index) => Number(value) !== nextData[index]);
    const currentCenterText = chartRef.current.options.plugins?.centerText?.text;
    const hasCenterTextChanged = currentCenterText !== centerText;

    if (hasDataChanged) {
      dataset.data = nextData;
    }

    if (hasCenterTextChanged && chartRef.current.options.plugins?.centerText) {
      chartRef.current.options.plugins.centerText.text = centerText;
    }

    if (hasDataChanged || hasCenterTextChanged) {
      chartRef.current.update();
    }

    return undefined;
  }, [centerText, safeNegative, safeNeutral, safePositive, total]);

  // Modified: debug logging stays dev-only and avoids repeated identical logs.
  useEffect(() => {
    if (
      import.meta.env.DEV &&
      typeof confidence === "number" &&
      Number.isFinite(confidence) &&
      lastLoggedConfidenceRef.current !== confidence
    ) {
      console.debug("Sentiment chart confidence:", confidence);
      lastLoggedConfidenceRef.current = confidence;
    }
  }, [confidence]);

  useEffect(() => {
    return () => {
      if (chartRef.current) {
        chartRef.current.canvas.style.cursor = "default";
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, []);

  // Modified: extra resize safety for responsive canvas rendering.
  useEffect(() => {
    const handleResize = () => {
      if (chartRef.current) {
        chartRef.current.resize();
      }
    };

    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  if (total === 0) {
    return (
      <div className="flex h-72 items-center justify-center rounded-3xl border border-white/10 bg-slate-900/40 px-6 text-center text-sm text-slate-300">
        No sentiment data is available yet.
      </div>
    );
  }

  return (
    <div
      className="h-72 rounded-3xl border border-white/10 bg-slate-900/40 p-4 shadow-[0_18px_60px_rgba(15,23,42,0.25)]"
      style={{
        opacity: isChartVisible ? 1 : 0,
        transition: "opacity 320ms ease",
      }}
    >
      <canvas
        ref={canvasRef}
        aria-label={ariaLabel}
        role="img"
        tabIndex={0}
        className="rounded-2xl focus:outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-emerald-400"
      />
    </div>
  );
}

export default SentimentChart;
