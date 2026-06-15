import { FaceDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");

const bpmValue = document.getElementById("bpmValue");
const qualityValue = document.getElementById("qualityValue");
const fpsValue = document.getElementById("fpsValue");
const statusValue = document.getElementById("statusValue");
const permissionModal = document.getElementById("permissionModal");
const permissionButton = document.getElementById("permissionButton");
const settingsToggle = document.getElementById("settingsToggle");
const settingsGrid = document.getElementById("settingsGrid");
const contentGrid = document.getElementById("contentGrid");
const columnResizer = document.getElementById("columnResizer");
const resolutionValue = document.getElementById("resolutionValue");

const windowSecInput = document.getElementById("windowSecInput");
const minBpmInput = document.getElementById("minBpmInput");
const maxBpmInput = document.getElementById("maxBpmInput");
const bpmIntervalInput = document.getElementById("bpmIntervalInput");
const cameraResolutionSelect = document.getElementById("cameraResolutionSelect");

const showRgbR = document.getElementById("showRgbR");
const showRgbG = document.getElementById("showRgbG");
const showRgbB = document.getElementById("showRgbB");
const showRppgRaw = document.getElementById("showRppgRaw");
const showRppgFiltered = document.getElementById("showRppgFiltered");
const showFaceBox = document.getElementById("showFaceBox");
const showRoiBoxes = document.getElementById("showRoiBoxes");

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
  cameraCapabilities: null,
  rppgPlotRaw: [],
  rppgPlotFiltered: [],
  lastRppgPlotTime: null,
  rppgYAxis: { min: -3, max: 3 },
};

const DEFAULT_RESAMPLE_FS = 30;
const DETECTION_INTERVAL_MS = 100;
const FACE_BOX_TTL_MS = 1000;
const QUALITY_THRESHOLD = 2.2;
const MAX_RGB_HISTORY = 300;
const RPPG_DISPLAY_SEC = 10;
const RPPG_DISPLAY_SAMPLES = DEFAULT_RESAMPLE_FS * RPPG_DISPLAY_SEC;
const CAMERA_RESOLUTION_PRESETS = [
  { label: "3840 x 2160", width: 3840, height: 2160 },
  { label: "2560 x 1440", width: 2560, height: 1440 },
  { label: "1920 x 1080", width: 1920, height: 1080 },
  { label: "1280 x 720", width: 1280, height: 720 },
  { label: "960 x 540", width: 960, height: 540 },
  { label: "640 x 480", width: 640, height: 480 },
  { label: "640 x 360", width: 640, height: 360 },
];

const rgbChart = createChart("rgbChart", [
  { label: "R", borderColor: "#ef4444" },
  { label: "G", borderColor: "#22c55e" },
  { label: "B", borderColor: "#3b82f6" },
]);

const rppgChart = createChart("rppgChart", [
  { label: "Raw rPPG", borderColor: "#94a3b8" },
  { label: "Pulse-like Bandpass", borderColor: "#06b6d4" },
]);

setupSeriesVisibilityControls();
setupCameraResolutionControl();
setupSettingsToggle();
setupColumnResizer();
setupStartupPermissionPrompt();

async function startApp() {
  if (appState.running) return;

  try {
    setStatus("初期化中...");
    await setupDetector();
    await setupCamera();
    resetStateForRun();
    appState.running = true;
    setStatus("計測中");
    requestAnimationFrame(processLoop);
  } catch (error) {
    console.error(error);
    setStatus(`初期化失敗: ${error.message}`);
    showPermissionModal();
  }
}

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
  appState.rppgPlotRaw = [];
  appState.rppgPlotFiltered = [];
  appState.lastRppgPlotTime = null;
  appState.rppgYAxis = { min: -3, max: 3 };
  updatePulseLabels(null, null);
  updateRgbChart();
  updateRppgChart([], []);
  updateResolutionLabel();
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

  const initialConstraints = buildCameraConstraints(cameraResolutionSelect?.value || "max");
  appState.stream = await navigator.mediaDevices.getUserMedia({
    video: initialConstraints,
    audio: false,
  });

  video.srcObject = appState.stream;

  const track = appState.stream.getVideoTracks()[0];
  appState.cameraCapabilities = track?.getCapabilities?.() || {};
  populateCameraResolutionOptions();
  await applySelectedCameraResolution();

  await video.play();
  syncCanvasSize();
  updateResolutionLabel();
}

function buildCameraConstraints(resolutionValue = "max") {
  const base = {
    facingMode: { ideal: "user" },
    frameRate: { ideal: 30 },
  };

  if (resolutionValue === "max") {
    return {
      ...base,
      width: { ideal: 99999 },
      height: { ideal: 99999 },
    };
  }

  const parsed = parseResolutionValue(resolutionValue);
  if (!parsed) return base;

  return {
    ...base,
    width: { ideal: parsed.width },
    height: { ideal: parsed.height },
  };
}

async function applySelectedCameraResolution() {
  const track = appState.stream?.getVideoTracks?.()[0];
  if (!track) return;

  const constraints = buildCameraConstraints(cameraResolutionSelect?.value || "max");
  const maxFrameRate = getCapabilityMax(appState.cameraCapabilities?.frameRate);
  if (maxFrameRate) {
    constraints.frameRate = { ideal: Math.min(30, maxFrameRate) };
  }

  try {
    await track.applyConstraints(constraints);
    await waitForVideoMetadata();
    syncCanvasSize();
    setStatus(appState.running ? "計測中" : "待機中");
  } catch (error) {
    console.warn("Could not apply camera resolution constraints.", error);
    setStatus(`解像度変更失敗: ${error.message}`);
  } finally {
    updateResolutionLabel();
  }
}

function parseResolutionValue(value) {
  const match = String(value || "").match(/^(\d+)x(\d+)$/);
  if (!match) return null;
  return { width: Number(match[1]), height: Number(match[2]) };
}

function waitForVideoMetadata() {
  if (video.readyState >= 1) return Promise.resolve();
  return new Promise((resolve) => {
    video.addEventListener("loadedmetadata", resolve, { once: true });
  });
}

function getCapabilityMax(capability) {
  if (!capability) return null;
  if (typeof capability.max === "number") return capability.max;
  if (Array.isArray(capability) && capability.length) return Math.max(...capability.filter(Number.isFinite));
  return null;
}

function stopApp() {
  appState.running = false;
  setStatus("停止中");

  if (appState.stream) {
    appState.stream.getTracks().forEach((track) => track.stop());
    appState.stream = null;
  }

  appState.cameraCapabilities = null;
  populateCameraResolutionOptions();
  video.srcObject = null;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  updateResolutionLabel();
}

function syncCanvasSize() {
  overlay.width = video.videoWidth || 640;
  overlay.height = video.videoHeight || 480;
  const videoStage = document.querySelector(".video-stage");
  if (videoStage && overlay.width && overlay.height) {
    videoStage.style.aspectRatio = `${overlay.width} / ${overlay.height}`;
  }
  updateResolutionLabel();

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
    updateResolutionLabel();

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

  return detections
    .map((detection) => ({
      boundingBox: detection.boundingBox,
      score: detection.categories?.[0]?.score ?? 0,
    }))
    .sort((a, b) => b.score - a.score)[0] ?? null;
}

function getMultiRois(faceBox) {
  if (!faceBox) return [];

  // Small ROIs are used instead of one large patch so that hair, eyes, mouth,
  // and specular highlights affect the pulse estimate less.
  const relBoxes = [
    [0.30, 0.12, 0.70, 0.27], // forehead
    [0.18, 0.30, 0.36, 0.48], // left temple / upper cheek
    [0.64, 0.30, 0.82, 0.48], // right temple / upper cheek
    [0.14, 0.48, 0.36, 0.68], // left cheek
    [0.64, 0.48, 0.86, 0.68], // right cheek
    [0.38, 0.46, 0.48, 0.66], // left side of nose
    [0.52, 0.46, 0.62, 0.66], // right side of nose
    [0.36, 0.72, 0.64, 0.84], // chin / lower face
  ];

  return relBoxes
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

  const drawScale = getOverlayDrawScale();
  const faceLineWidth = 3 * drawScale;
  const roiLineWidth = 2 * drawScale;
  const faceFontSize = 20 * drawScale;
  const roiFontSize = 15 * drawScale;
  const textPadding = 4 * drawScale;
  const shouldShowFaceBox = showFaceBox ? showFaceBox.checked : true;
  const shouldShowRoiBoxes = showRoiBoxes ? showRoiBoxes.checked : true;

  overlayCtx.lineJoin = "round";
  overlayCtx.lineCap = "round";

  if (faceBox && shouldShowFaceBox) {
    overlayCtx.strokeStyle = "#3b82f6";
    overlayCtx.lineWidth = faceLineWidth;
    overlayCtx.strokeRect(faceBox.originX, faceBox.originY, faceBox.width, faceBox.height);
    overlayCtx.fillStyle = "#3b82f6";
    overlayCtx.font = `700 ${faceFontSize}px Segoe UI, Arial, sans-serif`;
    const label = typeof faceScore === "number" ? `Face ${faceScore.toFixed(2)}` : "Face";
    overlayCtx.fillText(
      label,
      faceBox.originX,
      Math.max(faceFontSize, faceBox.originY - 6 * drawScale)
    );
  }

  if (shouldShowRoiBoxes) {
    overlayCtx.strokeStyle = "#ef4444";
    overlayCtx.lineWidth = roiLineWidth;
    overlayCtx.font = `700 ${roiFontSize}px Segoe UI, Arial, sans-serif`;
    roiBoxes.forEach((box, index) => {
      overlayCtx.strokeRect(box.x, box.y, box.width, box.height);
      overlayCtx.fillStyle = "#ef4444";
      overlayCtx.fillText(`${index + 1}`, box.x + textPadding, box.y + roiFontSize);
    });
  }
}

function getOverlayDrawScale() {
  const cssWidth = overlay.clientWidth || overlay.width || 1;
  const cssHeight = overlay.clientHeight || overlay.height || 1;
  const scaleX = overlay.width / cssWidth;
  const scaleY = overlay.height / cssHeight;
  return Math.max(scaleX, scaleY, 0.1);
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

  const raw = normalize(detrend(resampled.x, Math.round(DEFAULT_RESAMPLE_FS * 1.5)));
  let filtered = bandpassBiquad(raw, DEFAULT_RESAMPLE_FS, getMinBpm() / 60, getMaxBpm() / 60);
  filtered = normalize(movingAverage(filtered, 7));
  appendStableRppgSamples(resampled.t, raw, filtered);
  updateRppgChart();

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

function detrend(arr, windowSize) {
  if (arr.length === 0) return [];
  const trend = movingAverage(arr, Math.max(3, windowSize | 1));
  return arr.map((v, i) => v - trend[i]);
}

function bandpassBiquad(sig, fs, lowHz, highHz) {
  if (!sig.length) return [];
  const hp = biquadFilter(sig, makeHighpassCoeffs(lowHz, fs, 0.707));
  const lp = biquadFilter(hp, makeLowpassCoeffs(highHz, fs, 0.707));
  const reversed = biquadFilter([...lp].reverse(), makeHighpassCoeffs(lowHz, fs, 0.707));
  return biquadFilter(reversed, makeLowpassCoeffs(highHz, fs, 0.707)).reverse();
}

function makeHighpassCoeffs(freq, fs, q) {
  const w0 = (2 * Math.PI * freq) / fs;
  const cosW0 = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * q);
  let b0 = (1 + cosW0) / 2;
  let b1 = -(1 + cosW0);
  let b2 = (1 + cosW0) / 2;
  const a0 = 1 + alpha;
  let a1 = -2 * cosW0;
  let a2 = 1 - alpha;
  return normalizeBiquad({ b0, b1, b2, a0, a1, a2 });
}

function makeLowpassCoeffs(freq, fs, q) {
  const w0 = (2 * Math.PI * freq) / fs;
  const cosW0 = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * q);
  let b0 = (1 - cosW0) / 2;
  let b1 = 1 - cosW0;
  let b2 = (1 - cosW0) / 2;
  const a0 = 1 + alpha;
  let a1 = -2 * cosW0;
  let a2 = 1 - alpha;
  return normalizeBiquad({ b0, b1, b2, a0, a1, a2 });
}

function normalizeBiquad(c) {
  return {
    b0: c.b0 / c.a0,
    b1: c.b1 / c.a0,
    b2: c.b2 / c.a0,
    a1: c.a1 / c.a0,
    a2: c.a2 / c.a0,
  };
}

function biquadFilter(sig, c) {
  const out = new Array(sig.length);
  let x1 = 0;
  let x2 = 0;
  let y1 = 0;
  let y2 = 0;
  for (let i = 0; i < sig.length; i += 1) {
    const x0 = sig[i];
    const y0 = c.b0 * x0 + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
    out[i] = y0;
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
  }
  return out;
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
  applySeriesVisibility();
  rgbChart.update("none");
}

function appendStableRppgSamples(t, raw, filtered) {
  if (!t.length) return;

  let startIndex = 0;
  if (appState.lastRppgPlotTime != null) {
    startIndex = t.findIndex((time) => time > appState.lastRppgPlotTime + 1e-6);
    if (startIndex < 0) return;
  }

  for (let i = startIndex; i < t.length; i += 1) {
    appState.rppgPlotRaw.push(raw[i]);
    appState.rppgPlotFiltered.push(filtered[i]);
    appState.lastRppgPlotTime = t[i];
  }

  const keep = getRppgDisplaySamples() * 2;
  if (appState.rppgPlotRaw.length > keep) {
    appState.rppgPlotRaw = appState.rppgPlotRaw.slice(-keep);
    appState.rppgPlotFiltered = appState.rppgPlotFiltered.slice(-keep);
  }
}

function updateRppgChart() {
  const displayLen = getRppgDisplaySamples();
  const rawSeries = toFixedFlowSeries(appState.rppgPlotRaw, displayLen);
  const filteredSeries = toFixedFlowSeries(appState.rppgPlotFiltered, displayLen);

  rppgChart.data.labels = Array.from({ length: displayLen }, (_, i) => i + 1);
  rppgChart.data.datasets[0].data = rawSeries;
  rppgChart.data.datasets[1].data = filteredSeries;
  updateRppgYAxis(rawSeries, filteredSeries);
  applySeriesVisibility();
  rppgChart.update("none");
}

function updateRppgYAxis(...seriesList) {
  const visibleValues = seriesList.flat().filter((v) => Number.isFinite(v));
  if (visibleValues.length < 8) return;

  let min = Math.min(...visibleValues);
  let max = Math.max(...visibleValues);
  if (max - min < 0.5) {
    const center = (max + min) / 2;
    min = center - 0.25;
    max = center + 0.25;
  }

  const margin = (max - min) * 0.18;
  const targetMin = min - margin;
  const targetMax = max + margin;
  const current = appState.rppgYAxis;
  const growAlpha = 0.18;
  const shrinkAlpha = 0.035;

  current.min = smoothAxisLimit(current.min, targetMin, targetMin < current.min ? growAlpha : shrinkAlpha);
  current.max = smoothAxisLimit(current.max, targetMax, targetMax > current.max ? growAlpha : shrinkAlpha);

  rppgChart.options.scales.y.min = current.min;
  rppgChart.options.scales.y.max = current.max;
}

function smoothAxisLimit(prev, next, alpha) {
  if (!Number.isFinite(prev)) return next;
  return prev + (next - prev) * alpha;
}

function toFixedFlowSeries(values, displayLen) {
  const out = new Array(displayLen).fill(null);
  const tail = values.slice(-displayLen);
  const offset = displayLen - tail.length;
  for (let i = 0; i < tail.length; i += 1) {
    out[offset + i] = tail[i];
  }
  return out;
}

function getRppgDisplaySamples() {
  return Math.max(64, Math.round(getWindowSec() * DEFAULT_RESAMPLE_FS), RPPG_DISPLAY_SAMPLES);
}


function setupStartupPermissionPrompt() {
  showPermissionModal();
  permissionButton?.addEventListener("click", async () => {
    hidePermissionModal();
    await startApp();
  });
}

function showPermissionModal() {
  permissionModal?.classList.remove("hidden");
  requestAnimationFrame(() => permissionButton?.focus());
}

function hidePermissionModal() {
  permissionModal?.classList.add("hidden");
}

function setupSettingsToggle() {
  if (!settingsToggle || !settingsGrid) return;

  const toggle = () => {
    const panel = settingsToggle.closest(".settings-panel");
    const expanded = settingsToggle.getAttribute("aria-expanded") !== "false";
    settingsToggle.setAttribute("aria-expanded", String(!expanded));
    settingsGrid.hidden = expanded;
    panel?.classList.toggle("collapsed", expanded);
  };

  settingsToggle.addEventListener("click", toggle);
  settingsToggle.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      toggle();
    }
  });
}

function setupColumnResizer() {
  if (!contentGrid || !columnResizer) return;

  const resizeToClientX = (clientX) => {
    const rect = contentGrid.getBoundingClientRect();
    const minLeft = 240;
    const minRight = 240;
    const x = clamp(clientX - rect.left, minLeft, Math.max(minLeft, rect.width - minRight));
    const percent = (x / rect.width) * 100;
    contentGrid.style.setProperty("--camera-panel-width", `${percent.toFixed(2)}%`);
    rgbChart.resize();
    rppgChart.resize();
  };

  columnResizer.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    document.body.classList.add("resizing-columns");
    columnResizer.setPointerCapture?.(event.pointerId);

    const onMove = (moveEvent) => resizeToClientX(moveEvent.clientX);
    const onUp = () => {
      document.body.classList.remove("resizing-columns");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  });
}

function setupCameraResolutionControl() {
  populateCameraResolutionOptions();
  cameraResolutionSelect?.addEventListener("change", async () => {
    if (!appState.stream) return;
    setStatus("解像度変更中...");
    await applySelectedCameraResolution();
  });
}

function populateCameraResolutionOptions() {
  if (!cameraResolutionSelect) return;

  const previous = cameraResolutionSelect.value || "max";
  const capabilities = appState.cameraCapabilities || {};
  const maxWidth = getCapabilityMax(capabilities.width);
  const maxHeight = getCapabilityMax(capabilities.height);
  const candidates = [];

  if (maxWidth && maxHeight) {
    candidates.push({ label: `Max available (${maxWidth} x ${maxHeight})`, value: "max" });
  } else {
    candidates.push({ label: "Max available", value: "max" });
  }

  for (const preset of CAMERA_RESOLUTION_PRESETS) {
    if (maxWidth && preset.width > maxWidth) continue;
    if (maxHeight && preset.height > maxHeight) continue;
    candidates.push({ label: preset.label, value: `${preset.width}x${preset.height}` });
  }

  cameraResolutionSelect.innerHTML = "";
  for (const item of candidates) {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    cameraResolutionSelect.appendChild(option);
  }

  cameraResolutionSelect.value = candidates.some((item) => item.value === previous) ? previous : "max";
}

function setupSeriesVisibilityControls() {
  [showRgbR, showRgbG, showRgbB, showRppgRaw, showRppgFiltered, showFaceBox, showRoiBoxes]
    .filter(Boolean)
    .forEach((checkbox) => {
      checkbox.addEventListener("change", () => {
        applySeriesVisibility();
        rgbChart.update("none");
        rppgChart.update("none");
      });
    });
  applySeriesVisibility();
}

function applySeriesVisibility() {
  rgbChart.data.datasets[0].hidden = showRgbR ? !showRgbR.checked : false;
  rgbChart.data.datasets[1].hidden = showRgbG ? !showRgbG.checked : false;
  rgbChart.data.datasets[2].hidden = showRgbB ? !showRgbB.checked : false;
  rppgChart.data.datasets[0].hidden = showRppgRaw ? !showRppgRaw.checked : false;
  rppgChart.data.datasets[1].hidden = showRppgFiltered ? !showRppgFiltered.checked : false;
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
        spanGaps: false,
      })),
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { ticks: { maxTicksLimit: 6 } },
      },
    },
  });
}

function setStatus(text) {
  if (statusValue) statusValue.textContent = text;
}

function updateResolutionLabel() {
  if (!resolutionValue) return;

  const track = appState.stream?.getVideoTracks?.()[0];
  const settings = track?.getSettings?.() || {};
  const width = settings.width || video.videoWidth || overlay.width || 0;
  const height = settings.height || video.videoHeight || overlay.height || 0;

  resolutionValue.textContent = width && height ? `${width} x ${height}` : "--";
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
