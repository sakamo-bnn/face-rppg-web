import { initFaceDetector, detectFace } from "./face.js";
import { updateSignal } from "./signal.js";
import { drawOverlay, drawWaveform } from "./draw.js";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const waveform = document.getElementById("waveform");

const bpmEl = document.getElementById("bpm");
const statusEl = document.getElementById("status");
const fpsEl = document.getElementById("fps");

let stream = null;
let running = false;
let lastTime = performance.now();

function setStatus(text) {
    statusEl.textContent = text;
}

async function startCamera() {
    try {
        setStatus("カメラ起動中...");

        if (stream) {
            stopCamera();
        }

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        resizeCanvas();
        setStatus("カメラ起動完了");
        return true;
    } catch (err) {
        console.error("startCamera error:", err);

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

async function loop() {
    if (!running) return;

    requestAnimationFrame(loop);

    const now = performance.now();
    const dt = now - lastTime;
    lastTime = now;

    fpsEl.textContent = dt > 0 ? (1000 / dt).toFixed(1) : "--";

    if (video.readyState < 2) return;

    resizeCanvas();

    const face = await detectFace(video);
    const result = updateSignal(video, face);

    drawOverlay(overlay, face);
    drawWaveform(waveform, result?.signal || []);

    if (result && result.bpm) {
        bpmEl.textContent = result.bpm.toFixed(0);
    } else {
        bpmEl.textContent = "--";
    }

    setStatus(face ? "顔検出中" : "顔が見つかりません");
}

async function start() {
    if (running) return;

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

    running = true;
    lastTime = performance.now();
    setStatus("計測中");
    loop();
}

window.addEventListener("load", start);
window.addEventListener("beforeunload", stopCamera);
window.addEventListener("resize", resizeCanvas);

setStatus("待機中");