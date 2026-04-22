import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

// const { FaceDetector, FilesetResolver } = vision;
import { FaceDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const startButton = document.getElementById("startButton");

const bpmValue = document.getElementById("bpmValue");
const qualityValue = document.getElementById("qualityValue");
const fpsValue = document.getElementById("fpsValue");
const statusValue = document.getElementById("statusValue");

const windowSecInput = document.getElementById("windowSecInput");
const minBpmInput = document.getElementById("minBpmInput");
const maxBpmInput = document.getElementById("maxBpmInput");
const bpmIntervalInput = document.getElementById("bpmIntervalInput");

const appState = {
  detector: null,
  stream: null,
  running: false,
  lastFrameTime: 0,
  fpsSmooth: null,
  lastDetectionTs: 0,
  lastDetectionFoundAt: 0,
  lastDetectionBox: null,
  lastDetectionScore: null,
  lastBpmUpdateAt: 0,
  lastBpm: null,
  lastQuality: 0,
  rgbHistory: [],
  signalHistory: [],
};

const RGB_HISTORY_LENGTH = 300;
const DEFAULT_RESAMPLE_FS = 30;
const DETECTION_INTERVAL_MS = 100;
const FACE_BOX_TTL_MS = 1000;
const QUALITY_THRESHOLD = 2.2;

const rgbChart = createChart("rgbChart", [
  { label: "R", borderColor: "#ef4444" },
  { label: "G", borderColor: "#22c55e" },
  { label: "B", borderColor: "#3b82f6" },
]);

const rppgChart = createChart("rppgChart", [
  { label: "Raw", borderColor: "#94a3b8" },
  { label: "Filtered", borderColor: "#06b6d4" },
]);

startButton.addEventListener("click", async () => {
  if (appState.running) {
    stopApp();
    return;
  }

  try {
    setStatus("初期化中...");
    await setupDetector();
    await setupCamera();
    appState.running = true;
    startButton.textContent = "停止";
    setStatus("計測中");
    requestAnimationFrame(processLoop);
  } catch (error) {
    console.error(error);
    setStatus(`初期化失敗: ${error.message}`);
  }
});

async function setupDetector() {
  if (appState.detector) return;

  const visionFiles = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  appState.detector = await FaceDetector.createFromOptions(visionFiles, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    },
    runningMode: "VIDEO",
    minDetectionConfidence: 0.6,
  });
}

async function setupCamera() {
  if (appState.stream) return;

  appState.stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 30, max: 30 },
    },
    audio: false,
  });

  video.srcObject = appState.stream;
  await video.play();
  syncCanvasSize();
}

function stopApp() {
  appState.running = false;
  startButton.textContent = "計測開始";
  setStatus("停止中");

  if (appState.stream) {
    appState.stream.getTracks().forEach((track) => track.stop());
    appState.stream = null;
  }

  video.srcObject = null;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
}

function syncCanvasSize() {
  overlay.width = video.videoWidth || 640;
  overlay.height = video.videoHeight || 480;
}

async function processLoop() {
  if (!appState.running) return;

  const now = performance.now();
  updateFps(now);

  if (video.readyState >= 2) {
    if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
      syncCanvasSize();
    }

    let faceBox = appState.lastDetectionBox;
    let faceScore = appState.lastDetectionScore;

    if (now - appState.lastDetectionTs >= DETECTION_INTERVAL_MS) {
      const detection = appState.detector.detectForVideo(video, now);
      const best = pickBestFace(detection?.detections || []);
      appState.lastDetectionTs = now;

      if (best) {
        faceBox = best.boundingBox;
        faceScore = best.score;
        appState.lastDetectionBox = faceBox;
        appState.lastDetectionScore = faceScore;
        appState.lastDetectionFoundAt = now;
      } else if (now - appState.lastDetectionFoundAt > FACE_BOX_TTL_MS) {
        faceBox = null;
        faceScore = null;
        appState.lastDetectionBox = null;
        appState.lastDetectionScore = null;
      }
    }

    const roiBoxes = getMultiRois(faceBox);
    drawOverlay(faceBox, faceScore, roiBoxes);

    if (roiBoxes.length > 0) {
      const rgb = extractMeanRgb(video, roiBoxes);
      pushRgbSample(now / 1000, rgb);
      updateRgbChart();
      updatePulseEstimation(now / 1000);
      setStatus("計測中");
    } else {
      setStatus("顔が見つかりません");
      updatePulseLabels(null, null);
    }
  }

  requestAnimationFrame(processLoop);
}

function pickBestFace(detections) {
  if (!detections.length) return null;

  const sorted = detections
    .map((detection) => ({
      boundingBox: detection.boundingBox,
      score: detection.categories?.[0]?.score ?? 0,
    }))
    .sort((a, b) => b.score - a.score);

  return sorted[0] || null;
}

function getMultiRois(faceBox) {
  if (!faceBox) return [];

  const foreheadRel = [0.28, 0.14, 0.72, 0.3];
  const leftCheekRel = [0.12, 0.45, 0.34, 0.68];
  const rightCheekRel = [0.66, 0.45, 0.88, 0.68];
  return [foreheadRel, leftCheekRel, rightCheekRel]
    .map((rel) => roiFromRelBox(faceBox, rel))
    .filter(Boolean);
}

function roiFromRelBox(faceBox, rel) {
  const x1 = faceBox.originX + faceBox.width * rel[0];
  const y1 = faceBox.originY + faceBox.height * rel[1];
  const x2 = faceBox.originX + faceBox.width * rel[2];
  const y2 = faceBox.originY + faceBox.height * rel[3];

  if (x2 <= x1 || y2 <= y1) return null;

  return {
    x: Math.max(0, x1),
    y: Math.max(0, y1),
    width: Math.max(1, x2 - x1),
    height: Math.max(1, y2 - y1),
  };
}

function drawOverlay(faceBox, faceScore, roiBoxes) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  overlayCtx.save();
  overlayCtx.translate(overlay.width, 0);
  overlayCtx.scale(-1, 1);

  if (faceBox) {
    overlayCtx.strokeStyle = "#22c55e";
    overlayCtx.lineWidth = 3;
    overlayCtx.strokeRect(faceBox.originX, faceBox.originY, faceBox.width, faceBox.height);
    overlayCtx.fillStyle = "#22c55e";
    overlayCtx.font = "16px sans-serif";
    overlayCtx.fillText(`face ${faceScore.toFixed(2)}`, faceBox.originX, Math.max(18, faceBox.originY - 8));
  }

  overlayCtx.strokeStyle = "#ef4444";
  overlayCtx.lineWidth = 2;
  roiBoxes.forEach((box) => {
    overlayCtx.strokeRect(box.x, box.y, box.width, box.height);
  });
  overlayCtx.restore();
}

function extractMeanRgb(videoElement, roiBoxes) {
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = overlay.width;
  tempCanvas.height = overlay.height;
  const ctx = tempCanvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(videoElement, 0, 0, tempCanvas.width, tempCanvas.height);

  const allMeans = roiBoxes.map((box) => meanRgbInBox(ctx, box));
  const mean = allMeans.reduce(
    (acc, cur) => ({
      r: acc.r + cur.r,
      g: acc.g + cur.g,
      b: acc.b + cur.b,
    }),
    { r: 0, g: 0, b: 0 }
  );

  return {
    r: mean.r / allMeans.length,
    g: mean.g / allMeans.length,
    b: mean.b / allMeans.length,
  };
}

function meanRgbInBox(ctx, box) {
  const imageData = ctx.getImageData(box.x, box.y, box.width, box.height).data;
  let r = 0;
  let g = 0;
  let b = 0;
  let count = 0;

  for (let i = 0; i < imageData.length; i += 4) {
    r += imageData[i];
    g += imageData[i + 1];
    b += imageData[i + 2];
    count += 1;
  }

  return {
    r: r / count,
    g: g / count,
    b: b / count,
  };
}

function pushRgbSample(ts, rgb) {
  const windowSec = Number(windowSecInput.value);
  appState.rgbHistory.push({ ts, ...rgb });
  appState.signalHistory.push({ ts, rgb });

  if (appState.rgbHistory.length > RGB_HISTORY_LENGTH) {
    appState.rgbHistory.shift();
  }

  while (appState.signalHistory.length > 0 && ts - appState.signalHistory[0].ts > windowSec) {
    appState.signalHistory.shift();
  }
}

function updateRgbChart() {
  const labels = appState.rgbHistory.map((_, index) => index + 1);
  rgbChart.data.labels = labels;
  rgbChart.data.datasets[0].data = appState.rgbHistory.map((p) => p.r);
  rgbChart.data.datasets[1].data = appState.rgbHistory.map((p) => p.g);
  rgbChart.data.datasets[2].data = appState.rgbHistory.map((p) => p.b);
  rgbChart.update("none");
}

function updatePulseEstimation(nowSec) {
  if (appState.signalHistory.length < 64) {
    rppgChart.data.labels = [];
    rppgChart.data.datasets.forEach((dataset) => {
      dataset.data = [];
    });
    rppgChart.update("none");
    updatePulseLabels(appState.lastBpm, appState.lastQuality);
    return;
  }

  const samples = appState.signalHistory.map((item) => [item.rgb.r, item.rgb.g, item.rgb.b]);
  const timestamps = appState.signalHistory.map((item) => item.ts);

  const chromSignal = extractChromSignal(samples);
  const resampled = resampleSignal(timestamps, chromSignal, DEFAULT_RESAMPLE_FS);
  if (!resampled) {
    setStatus("補間失敗");
    return;
  }

  let filtered = normalize(resampled.values);
  filtered = movingAverage(filtered, 5);
  filtered = bandLimitByFft(
    filtered,
    DEFAULT_RESAMPLE_FS,
    Number(minBpmInput.value) / 60,
    Number(maxBpmInput.value) / 60
  );

  const raw = normalize(resampled.values);
  rppgChart.data.labels = filtered.map((_, index) => index + 1);
  rppgChart.data.datasets[0].data = raw;
  rppgChart.data.datasets[1].data = filtered;
  rppgChart.update("none");

  const updateIntervalSec = Number(bpmIntervalInput.value);
  if (nowSec - appState.lastBpmUpdateAt >= updateIntervalSec) {
    const result = estimateBpmFromFft(
      filtered,
      DEFAULT_RESAMPLE_FS,
      Number(minBpmInput.value),
      Number(maxBpmInput.value)
    );

    if (result && result.quality >= QUALITY_THRESHOLD) {
      appState.lastBpm = smoothBpm(appState.lastBpm, result.bpm, 0.08);
      appState.lastQuality = result.quality;
    }
    appState.lastBpmUpdateAt = nowSec;
  }

  updatePulseLabels(appState.lastBpm, appState.lastQuality);
}

function extractChromSignal(rgbBuffer) {
  if (rgbBuffer.length < 2) return [];

  const mean = rgbBuffer.reduce(
    (acc, [r, g, b]) => [acc[0] + r, acc[1] + g, acc[2] + b],
    [0, 0, 0]
  ).map((v) => v / rgbBuffer.length || 1);

  const normalized = rgbBuffer.map(([r, g, b]) => [r / mean[0], g / mean[1], b / mean[2]]);
  const xsig = normalized.map(([r, g]) => 3 * r - 2 * g);
  const ysig = normalized.map(([r, g, b]) => 1.5 * r + g - 1.5 * b);
  const stdY = standardDeviation(ysig);
  const alpha = stdY < 1e-8 ? 0 : standardDeviation(xsig) / stdY;

  const signal = xsig.map((x, index) => x - alpha * ysig[index]);
  const meanSignal = average(signal);
  return signal.map((value) => value - meanSignal);
}

function resampleSignal(timestamps, values, targetFs) {
  if (timestamps.length < 2 || values.length < 2) return null;

  const t0 = timestamps[0];
  const t1 = timestamps[timestamps.length - 1];
  if (t1 <= t0) return null;

  const step = 1 / targetFs;
  const outputTs = [];
  const outputValues = [];

  for (let t = t0; t < t1; t += step) {
    outputTs.push(t);
    outputValues.push(linearInterpolate(timestamps, values, t));
  }

  if (outputValues.length < 8) return null;
  return { timestamps: outputTs, values: outputValues };
}

function linearInterpolate(xs, ys, x) {
  let i = 0;
  while (i < xs.length - 1 && xs[i + 1] < x) i += 1;
  const x0 = xs[i];
  const x1 = xs[Math.min(i + 1, xs.length - 1)];
  const y0 = ys[i];
  const y1 = ys[Math.min(i + 1, ys.length - 1)];

  if (x1 === x0) return y0;
  const ratio = (x - x0) / (x1 - x0);
  return y0 + ratio * (y1 - y0);
}

function bandLimitByFft(signal, fs, minHz, maxHz) {
  const n = signal.length;
  const spectrum = dft(signal);

  for (let k = 0; k < spectrum.length; k += 1) {
    const freq = (k * fs) / n;
    if (freq < minHz || freq > maxHz) {
      spectrum[k].re = 0;
      spectrum[k].im = 0;
    }
  }

  return idft(spectrum).map((value) => Number.isFinite(value) ? value : 0);
}

function estimateBpmFromFft(signal, fs, minBpm, maxBpm) {
  const n = signal.length;
  if (n < 64 || fs <= 0) return null;

  const centered = signal.map((value) => value - average(signal));
  if (standardDeviation(centered) < 1e-6) return null;

  const windowed = centered.map((value, index) => value * hamming(index, n));
  const spectrum = dft(windowed).slice(0, Math.floor(n / 2));
  const amplitudes = spectrum.map(({ re, im }) => Math.hypot(re, im));

  const minHz = minBpm / 60;
  const maxHz = maxBpm / 60;
  const candidates = amplitudes
    .map((amp, index) => ({ freq: (index * fs) / n, amp }))
    .filter((item) => item.freq >= minHz && item.freq <= maxHz);

  if (candidates.length === 0) return null;

  const best = candidates.reduce((acc, cur) => (cur.amp > acc.amp ? cur : acc));
  const median = medianValue(candidates.map((item) => item.amp)) + 1e-8;

  return {
    bpm: best.freq * 60,
    quality: best.amp / median,
  };
}

function smoothBpm(prev, next, alpha) {
  if (next == null) return prev;
  if (prev == null) return next;
  return (1 - alpha) * prev + alpha * next;
}

function updatePulseLabels(bpm, quality) {
  bpmValue.textContent = bpm == null ? "--" : bpm.toFixed(1);
  qualityValue.textContent = quality == null ? "--" : quality.toFixed(2);
}

function updateFps(now) {
  if (!appState.lastFrameTime) {
    appState.lastFrameTime = now;
    return;
  }

  const dt = (now - appState.lastFrameTime) / 1000;
  appState.lastFrameTime = now;
  if (dt <= 0) return;

  const fps = 1 / dt;
  appState.fpsSmooth = appState.fpsSmooth == null ? fps : appState.fpsSmooth * 0.9 + fps * 0.1;
  fpsValue.textContent = appState.fpsSmooth.toFixed(1);
}

function createChart(canvasId, datasetConfigs) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: datasetConfigs.map((config) => ({
        ...config,
        data: [],
        pointRadius: 0,
        borderWidth: 2,
        tension: 0.2,
      })),
    },
    options: {
      responsive: true,
      animation: false,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148, 163, 184, 0.12)" },
        },
        y: {
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148, 163, 184, 0.12)" },
        },
      },
      plugins: {
        legend: {
          labels: { color: "#e5e7eb" },
        },
      },
    },
  });
}

function setStatus(message) {
  statusValue.textContent = message;
}

function normalize(values) {
  const mean = average(values);
  const std = standardDeviation(values);
  if (std < 1e-8) return values.map(() => 0);
  return values.map((value) => (value - mean) / std);
}

function movingAverage(values, width) {
  if (values.length < width) return [...values];
  const half = Math.floor(width / 2);
  return values.map((_, index) => {
    const start = Math.max(0, index - half);
    const end = Math.min(values.length, index + half + 1);
    return average(values.slice(start, end));
  });
}

function average(values) {
  return values.reduce((acc, cur) => acc + cur, 0) / values.length;
}

function standardDeviation(values) {
  const mean = average(values);
  const variance = average(values.map((value) => (value - mean) ** 2));
  return Math.sqrt(variance);
}

function medianValue(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function hamming(index, length) {
  if (length <= 1) return 1;
  return 0.54 - 0.46 * Math.cos((2 * Math.PI * index) / (length - 1));
}

function dft(signal) {
  const n = signal.length;
  const output = [];

  for (let k = 0; k < n; k += 1) {
    let re = 0;
    let im = 0;
    for (let t = 0; t < n; t += 1) {
      const angle = (-2 * Math.PI * t * k) / n;
      re += signal[t] * Math.cos(angle);
      im += signal[t] * Math.sin(angle);
    }
    output.push({ re, im });
  }

  return output;
}

function idft(spectrum) {
  const n = spectrum.length;
  const output = [];

  for (let t = 0; t < n; t += 1) {
    let re = 0;
    for (let k = 0; k < n; k += 1) {
      const angle = (2 * Math.PI * t * k) / n;
      re += spectrum[k].re * Math.cos(angle) - spectrum[k].im * Math.sin(angle);
    }
    output.push(re / n);
  }

  return output;
}

window.addEventListener("resize", syncCanvasSize);
