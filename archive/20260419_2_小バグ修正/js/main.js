import { initFaceDetector, detectFace } from "./face.js";
import { updateSignal, resetSignal } from "./signal.js";
import { drawOverlay, drawWaveform, clearOverlay, clearWaveform } from "./draw.js";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const waveform = document.getElementById("waveform");

const bpmEl = document.getElementById("bpm");
const statusEl = document.getElementById("status");
const fpsEl = document.getElementById("fps");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

let stream = null;
let cameraReady = false;
let running = false;
let rafId = null;
let lastTime = performance.now();

function setStatus(text) {
    statusEl.textContent = text;
}

function setBpmText(value) {
    bpmEl.textContent = value;
}

function setFpsText(value) {
    fpsEl.textContent = value;
}

async function startCamera() {
    if (cameraReady && stream) {
        return true;
    }

    try {
        setStatus("カメラ起動中...");

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                // width: { ideal: 640 },
                // height: { ideal: 480 }
            },
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        const videoTrack = stream.getVideoTracks()[0];
        const trackSettings = videoTrack ? videoTrack.getSettings() : null;

        // console.log("[Camera] requested ideal:", {
        //     width: 640,
        //     height: 480,
        //     facingMode: "user"
        // });

        console.log("[Camera] actual track settings:", trackSettings);

        console.log("[Camera] actual video size:", {
            videoWidth: video.videoWidth,
            videoHeight: video.videoHeight
        });

        resizeCanvas();

        cameraReady = true;
        setStatus("カメラ起動完了");
        return true;
    } catch (err) {
        console.error("startCamera error:", err);

        cameraReady = false;

        if (err.name === "NotReadableError") {
            setStatus("カメラ使用中または起動不可");
        } else if (err.name === "NotAllowedError") {
            setStatus("カメラ権限が拒否されています");
        } else if (err.name === "NotFoundError") {
            setStatus("利用可能なカメラが見つかりません");
        } else {
            setStatus(`カメラ起動失敗: ${err.name}`);
        }

        return false;
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
    }
    cameraReady = false;
}

function resizeCanvas() {
    const w = video.videoWidth;
    const h = video.videoHeight;

    if (!w || !h) return;

    overlay.width = w;
    overlay.height = h;

    waveform.width = waveform.clientWidth;
    waveform.height = waveform.clientHeight;
}

function resetMeasurementView() {
    resetSignal();
    setBpmText("--");
    setFpsText("--");
    clearOverlay(overlay);
    clearWaveform(waveform);
}

async function loop() {
    if (!running) return;

    rafId = requestAnimationFrame(loop);

    const now = performance.now();
    const dt = now - lastTime;
    lastTime = now;

    setFpsText(dt > 0 ? (1000 / dt).toFixed(1) : "--");

    if (!cameraReady || video.readyState < 2) {
        return;
    }

    resizeCanvas();

    const face = await detectFace(video);
    const result = updateSignal(video, face);

    drawOverlay(overlay, face);
    drawWaveform(waveform, result?.signal || []);

    if (result && result.bpm) {
        setBpmText(result.bpm.toFixed(0));
    } else {
        setBpmText("--");
    }

    setStatus(face ? "計測中" : "顔が見つかりません");
}

async function startMeasurement() {
    const ok = await startCamera();
    if (!ok) return;

    if (running) {
        stopMeasurement();
    }

    resetMeasurementView();

    running = true;
    lastTime = performance.now();
    setStatus("計測開始");
    loop();
}

function stopMeasurement() {
    running = false;

    if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
    }

    setStatus("計測停止");
}

async function initializeApp() {
    setStatus("初期化中...");

    const ok = await startCamera();
    if (!ok) return;

    setStatus("顔検出モデル初期化中...");

    try {
        await initFaceDetector();
    } catch (err) {
        console.error("initFaceDetector error:", err);
        setStatus("顔検出モデル初期化失敗");
        return;
    }

    resizeCanvas();
    clearOverlay(overlay);
    clearWaveform(waveform);
    setBpmText("--");
    setFpsText("--");
    setStatus("カメラ待機中");
}

startBtn.addEventListener("click", startMeasurement);
stopBtn.addEventListener("click", stopMeasurement);

window.addEventListener("load", initializeApp);
window.addEventListener("beforeunload", stopCamera);
window.addEventListener("resize", resizeCanvas);