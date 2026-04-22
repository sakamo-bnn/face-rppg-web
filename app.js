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
  fpsSmooth: null,
  lastFrameTime: 0,
  lastDetectionTs: 0,
  lastDetectionFoundAt: 0,
  lastDetectionBox: null,
  lastDetectionScore: null,
  lastBpmUpdateAt: 0,
  lastBpm: null,
  lastQuality: 0,
  samples: [],
  offscreenCanvas: null,
  offscreenCtx: null,
};

const DEFAULT_RESAMPLE_FS = 30;
const DETECTION_INTERVAL_MS = 100;
const FACE_BOX_TTL_MS = 1000;
const QUALITY_THRESHOLD = 2.2;
const MAX_RGB_HISTORY = 300;

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
    resetStateForRun();
    appState.running = true;
    startButton.textContent = "停止";
    setStatus("計測中");
    requestAnimationFrame(processLoop);
  } catch (error) {
    console.error(error);
    setStatus(`初期化失敗: ${error.message}`);
  }
});

function resetStateForRun() {
  appState.fpsSmooth = null;
  appState.lastFrameTime = 0;
  appState.lastDetectionTs = 0;
  appState.lastDetectionFoundAt = 0;
  appState.lastDetectionBox = null;
  appState.lastDetectionScore = null;
  appState.lastBpmUpdateAt = 0;
  appState.lastBpm = null;
  appState.lastQuality = 0;
  appState.samples = [];
  updatePulseLabels(null, null);
  updateRgbChart();
  updateRppgChart([], []);
}

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

  if (!appState.offscreenCanvas) {
    appState.offscreenCanvas = document.createElement("canvas");
    appState.offscreenCtx = appState.offscreenCanvas.getContext("2d", { willReadFrequently: true });
  }
  appState.offscreenCanvas.width = overlay.width;
  appState.offscreenCanvas.height = overlay.height;
}

function processLoop(now) {
  if (!appState.running) return;

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
      updateRppgChart([], []);
    }
  }

  requestAnimationFrame(processLoop);
}

function pickBestFace(detections) {
  if (!detections.length) return null;

  return detections
    .map((detection) => ({
      boundingBox: detection.boundingBox,
      score: detection.categories?.[0]?.score ?? 0,
    }))
    .sort((a, b) => b.score - a.score)[0] ?? null;
}

function getMultiRois(faceBox) {
  if (!faceBox) return [];

  const foreheadRel = [0.28, 0.14, 0.72, 0.30];
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
    x: clamp(x1, 0, overlay.width - 1),
    y: clamp(y1, 0, overlay.height - 1),
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
    if (typeof faceScore === "number") {
      overlayCtx.fillText(`face ${faceScore.toFixed(2)}`, faceBox.originX, Math.max(18, faceBox.originY - 8));
    }
  }

  overlayCtx.strokeStyle = "#ef4444";
  overlayCtx.lineWidth = 2;
  roiBoxes.forEach((box) => {
    overlayCtx.strokeRect(box.x, box.y, box.width, box.height);
  });
  overlayCtx.restore();
}

function extractMeanRgb(videoElement, roiBoxes) {
  const canvas = appState.offscreenCanvas;
  const ctx = appState.offscreenCtx;
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

  const means = roiBoxes.map((box) => meanRgbInBox(ctx, box));
  const sum = means.reduce(
    (acc, cur) => ({ r: acc.r + cur.r, g: acc.g + cur.g, b: acc.b + cur.b }),
    { r: 0, g: 0, b: 0 }
  );

  return {
    r: sum.r / means.length,
    g: sum.g / means.length,
    b: sum.b / means.length,
  };
}

function meanRgbInBox(ctx, box) {
  const x = Math.floor(box.x);
  const y = Math.floor(box.y);
  const w = Math.max(1, Math.floor(box.width));
  const h = Math.max(1, Math.floor(box.height));
  const imageData = ctx.getImageData(x, y, w, h).data;

  let r = 0;
  let g = 0;
  let b = 0;
  const n = imageData.length / 4;

  for (let i = 0; i < imageData.length; i += 4) {
    r += imageData[i];
    g += imageData[i + 1];
    b += imageData[i + 2];
  }

  return { r: r / n, g: g / n, b: b / n };
}

function pushRgbSample(t, rgb) {
  appState.samples.push({ t, r: rgb.r, g: rgb.g, b: rgb.b });

  const windowSec = getWindowSec();
  while (appState.samples.length > 0 && t - appState.samples[0].t > windowSec) {
    appState.samples.shift();
  }
}

function updatePulseEstimation(nowSec) {
  if (appState.samples.length < 64) {
    updatePulseLabels(appState.lastBpm, appState.lastQuality);
    return;
  }

  const rgbSeq = appState.samples.map((s) => [s.r, s.g, s.b]);
  const tSeq = appState.samples.map((s) => s.t);

  const chromSig = extractChromSignal(rgbSeq);
  const resampled = resampleSignal(tSeq, chromSig, DEFAULT_RESAMPLE_FS);
  if (!resampled || resampled.x.length < 64) {
    updatePulseLabels(appState.lastBpm, appState.lastQuality);
    return;
  }

  let raw = normalize(resampled.x);
  let filtered = bandpassFft(raw, DEFAULT_RESAMPLE_FS, getMinBpm() / 60, getMaxBpm() / 60);
  filtered = movingAverage(filtered, 5);
  updateRppgChart(raw, filtered);

  const intervalSec = getBpmInterval();
  if (nowSec - appState.lastBpmUpdateAt < intervalSec) {
    updatePulseLabels(appState.lastBpm, appState.lastQuality);
    return;
  }

  const estimate = estimateBpmFromFft(filtered, DEFAULT_RESAMPLE_FS, getMinBpm(), getMaxBpm());
  if (estimate && estimate.quality >= QUALITY_THRESHOLD) {
    appState.lastBpm = smoothBpm(appState.lastBpm, estimate.bpm, 0.08);
    appState.lastQuality = estimate.quality;
  }

  appState.lastBpmUpdateAt = nowSec;
  updatePulseLabels(appState.lastBpm, appState.lastQuality);
}

function extractChromSignal(rgbBuffer) {
  const n = rgbBuffer.length;
  if (n < 2) return new Array(n).fill(0);

  const mean = [0, 0, 0];
  for (const [r, g, b] of rgbBuffer) {
    mean[0] += r;
    mean[1] += g;
    mean[2] += b;
  }
  mean[0] = mean[0] / n || 1;
  mean[1] = mean[1] / n || 1;
  mean[2] = mean[2] / n || 1;

  const xs = [];
  const ys = [];
  for (const [r, g, b] of rgbBuffer) {
    const R = r / mean[0];
    const G = g / mean[1];
    const B = b / mean[2];
    xs.push(3 * R - 2 * G);
    ys.push(1.5 * R + G - 1.5 * B);
  }

  const stdY = std(ys);
  const alpha = stdY < 1e-8 ? 0 : std(xs) / stdY;
  const out = xs.map((x, i) => x - alpha * ys[i]);
  const m = average(out);
  return out.map((v) => v - m);
}

function resampleSignal(t, x, targetFs) {
  if (t.length < 2 || x.length < 2) return null;
  const t0 = t[0];
  const t1 = t[t.length - 1];
  if (t1 <= t0) return null;

  const dt = 1 / targetFs;
  const tu = [];
  const xu = [];
  let j = 0;

  for (let tt = t0; tt < t1; tt += dt) {
    while (j < t.length - 2 && t[j + 1] < tt) j += 1;
    const ta = t[j];
    const tb = t[j + 1];
    const xa = x[j];
    const xb = x[j + 1];
    const ratio = tb === ta ? 0 : (tt - ta) / (tb - ta);
    tu.push(tt);
    xu.push(xa + ratio * (xb - xa));
  }

  return tu.length >= 8 ? { t: tu, x: xu } : null;
}

function estimateBpmFromFft(sig, fs, minBpm, maxBpm) {
  const n = sig.length;
  if (n < 64 || fs <= 0) return null;

  const centered = sig.map((v) => v - average(sig));
  if (std(centered) < 1e-6) return null;

  const windowed = centered.map((v, i) => v * hamming(i, n));
  const spectrum = rfftMagnitude(windowed);
  const freqs = spectrum.map((_, i) => (i * fs) / n);

  const minHz = minBpm / 60;
  const maxHz = maxBpm / 60;
  const selected = [];
  for (let i = 0; i < spectrum.length; i += 1) {
    if (freqs[i] >= minHz && freqs[i] <= maxHz) {
      selected.push({ freq: freqs[i], amp: spectrum[i] });
    }
  }
  if (!selected.length) return null;

  let peak = selected[0];
  for (const item of selected) {
    if (item.amp > peak.amp) peak = item;
  }

  const amps = selected.map((s) => s.amp).sort((a, b) => a - b);
  const median = amps[Math.floor(amps.length / 2)] + 1e-8;
  return { bpm: peak.freq * 60, quality: peak.amp / median };
}

function bandpassFft(sig, fs, lowHz, highHz) {
  const n = sig.length;
  const out = new Array(n).fill(0);
  for (let k = 0; k <= Math.floor(n / 2); k += 1) {
    const freq = (k * fs) / n;
    if (freq < lowHz || freq > highHz) continue;
    const { re, im } = dftBin(sig, k);
    for (let t = 0; t < n; t += 1) {
      const angle = (2 * Math.PI * k * t) / n;
      out[t] += (re * Math.cos(angle) - im * Math.sin(angle)) / n;
      if (k > 0 && k < n / 2) {
        out[t] += (re * Math.cos(angle) + im * Math.sin(angle)) / n;
      }
    }
  }
  return normalize(out);
}

function dftBin(sig, k) {
  const n = sig.length;
  let re = 0;
  let im = 0;
  for (let t = 0; t < n; t += 1) {
    const angle = (-2 * Math.PI * k * t) / n;
    re += sig[t] * Math.cos(angle);
    im += sig[t] * Math.sin(angle);
  }
  return { re, im };
}

function rfftMagnitude(sig) {
  const n = sig.length;
  const out = [];
  for (let k = 0; k <= Math.floor(n / 2); k += 1) {
    const { re, im } = dftBin(sig, k);
    out.push(Math.hypot(re, im));
  }
  return out;
}

function movingAverage(arr, size) {
  if (arr.length === 0 || size <= 1) return [...arr];
  const half = Math.floor(size / 2);
  return arr.map((_, i) => {
    let sum = 0;
    let count = 0;
    for (let j = i - half; j <= i + half; j += 1) {
      if (j >= 0 && j < arr.length) {
        sum += arr[j];
        count += 1;
      }
    }
    return sum / count;
  });
}

function normalize(arr) {
  const mean = average(arr);
  const sigma = std(arr);
  if (sigma < 1e-8) return arr.map((v) => v - mean);
  return arr.map((v) => (v - mean) / sigma);
}

function smoothBpm(prev, next, alpha) {
  if (next == null) return prev;
  if (prev == null) return next;
  return (1 - alpha) * prev + alpha * next;
}

function average(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function std(arr) {
  if (!arr.length) return 0;
  const mean = average(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length);
}

function hamming(i, n) {
  return 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1));
}

function updateFps(now) {
  if (!appState.lastFrameTime) {
    appState.lastFrameTime = now;
    return;
  }
  const dt = now - appState.lastFrameTime;
  appState.lastFrameTime = now;
  if (dt <= 0) return;

  const fps = 1000 / dt;
  appState.fpsSmooth = appState.fpsSmooth == null ? fps : 0.9 * appState.fpsSmooth + 0.1 * fps;
  fpsValue.textContent = appState.fpsSmooth.toFixed(1);
}

function updatePulseLabels(bpm, quality) {
  bpmValue.textContent = bpm == null ? "--" : bpm.toFixed(1);
  qualityValue.textContent = quality == null ? "--" : quality.toFixed(2);
}

function updateRgbChart() {
  const samples = appState.samples.slice(-MAX_RGB_HISTORY);
  rgbChart.data.labels = samples.map((_, i) => i + 1);
  rgbChart.data.datasets[0].data = samples.map((s) => s.r);
  rgbChart.data.datasets[1].data = samples.map((s) => s.g);
  rgbChart.data.datasets[2].data = samples.map((s) => s.b);
  rgbChart.update("none");
}

function updateRppgChart(raw, filtered) {
  const len = Math.max(raw.length, filtered.length);
  rppgChart.data.labels = Array.from({ length: len }, (_, i) => i + 1);
  rppgChart.data.datasets[0].data = raw;
  rppgChart.data.datasets[1].data = filtered;
  rppgChart.update("none");
}

function createChart(canvasId, datasets) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: datasets.map((dataset) => ({
        ...dataset,
        data: [],
        fill: false,
        tension: 0.2,
        pointRadius: 0,
        borderWidth: 2,
      })),
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: true } },
      scales: {
        x: { display: false },
        y: { ticks: { maxTicksLimit: 6 } },
      },
    },
  });
}

function setStatus(text) {
  statusValue.textContent = text;
}

function getWindowSec() {
  return Math.max(5, Number(windowSecInput.value) || 10);
}

function getMinBpm() {
  return Math.max(30, Number(minBpmInput.value) || 45);
}

function getMaxBpm() {
  return Math.max(getMinBpm() + 1, Number(maxBpmInput.value) || 180);
}

function getBpmInterval() {
  return Math.max(0.5, Number(bpmIntervalInput.value) || 1.0);
}

function clamp(v, min, max) {
  return Math.min(max, Math.max(min, v));
}
