// ===== import =====
import { detectFace } from "./face.js";
import { updateSignal } from "./signal.js";
import { drawOverlay, drawWaveform } from "./draw.js";

// ===== DOM =====
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const waveform = document.getElementById("waveform");

const bpmEl = document.getElementById("bpm");
const statusEl = document.getElementById("status");
const fpsEl = document.getElementById("fps");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

// ===== 状態 =====
let stream = null;
let running = false;
let lastTime = performance.now();

// ===== 初期化 =====
function setStatus(text) {
    statusEl.textContent = text;
}

// ===== カメラ起動 =====
async function startCamera() {
    try {
        setStatus("カメラ起動中...");


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


    } catch (err) {
        console.error(err);
        setStatus("カメラ起動失敗");
    }
}

// ===== カメラ停止 =====
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

// ===== Canvasサイズ同期 =====
function resizeCanvas() {
    const w = video.videoWidth;
    const h = video.videoHeight;

    overlay.width = w;
    overlay.height = h;

    waveform.width = waveform.clientWidth;
    waveform.height = waveform.clientHeight;
}

// ===== メインループ =====
function loop() {
    if (!running) return;

    requestAnimationFrame(loop);

    const now = performance.now();
    const dt = now - lastTime;
    lastTime = now;

    const fps = (1000 / dt).toFixed(1);
    fpsEl.textContent = fps;

    if (video.readyState < 2) return;

    // ===== 顔検出 =====
    const face = detectFace(video);

    // ===== 信号処理 =====
    const result = updateSignal(video, face);

    // ===== 描画 =====
    drawOverlay(overlay, face);
    drawWaveform(waveform, result?.signal || []);

    // ===== BPM表示 =====
    if (result && result.bpm) {
        bpmEl.textContent = result.bpm.toFixed(0);
    } else {
        bpmEl.textContent = "--";
    }
}

// ===== スタート =====
async function start() {
    if (running) return;

    await startCamera();

    running = true;
    lastTime = performance.now();

    setStatus("計測中");
    loop();
}

// ===== ストップ =====
function stop() {
    running = false;
    stopCamera();
    setStatus("停止中");
}

// ===== イベント =====
startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

// ===== 初期状態 =====
setStatus("待機中");
