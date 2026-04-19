// signal.js
// 役割:
// - ROI内の平均G値を取得
// - 時系列バッファに蓄積
// - 簡易的なBPM推定

const MAX_BUFFER_SIZE = 300; // 約10秒分（30fps想定）

let signalBuffer = [];
let timeBuffer = [];

// ===== メイン関数 =====
export function updateSignal(video, face) {
    if (!face || !face.roiBox) {
        return { signal: signalBuffer, bpm: null };
    }

    const roi = face.roiBox;

    // ===== ROIから平均G値を取得 =====
    const g = getAverageG(video, roi);

    const now = performance.now();

    // ===== バッファ追加 =====
    signalBuffer.push(g);
    timeBuffer.push(now);

    if (signalBuffer.length > MAX_BUFFER_SIZE) {
        signalBuffer.shift();
        timeBuffer.shift();
    }

    // ===== 正規化（簡易）=====
    const normalized = normalize(signalBuffer);

    // ===== BPM推定 =====
    const bpm = estimateBPM(normalized, timeBuffer);

    return {
        signal: normalized,
        bpm: bpm
    };
}

// ===== ROI内の平均G値 =====
function getAverageG(video, roi) {
    const canvas = getTempCanvas(video);
    const ctx = canvas.getContext("2d");

    const x = Math.floor(roi.x);
    const y = Math.floor(roi.y);
    const w = Math.floor(roi.width);
    const h = Math.floor(roi.height);

    if (w <= 0 || h <= 0) return 0;

    const imageData = ctx.getImageData(x, y, w, h);
    const data = imageData.data;

    let sumG = 0;
    let count = 0;

    for (let i = 0; i < data.length; i += 4) {
        const g = data[i + 1]; // G成分
        sumG += g;
        count++;
    }

    return count > 0 ? sumG / count : 0;
}

// ===== 一時Canvas（毎回生成しない）=====
let tempCanvas = null;

function getTempCanvas(video) {
    if (!tempCanvas) {
        tempCanvas = document.createElement("canvas");
    }

    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;

    const ctx = tempCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

    return tempCanvas;
}

// ===== 正規化 =====
function normalize(arr) {
    if (arr.length === 0) return [];

    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const centered = arr.map(v => v - mean);

    const max = Math.max(...centered.map(v => Math.abs(v))) || 1;

    return centered.map(v => v / max);
}

// ===== BPM推定（簡易ピーク法）=====
function estimateBPM(signal, time) {
    if (signal.length < 30) return null;

    const peaks = [];

    for (let i = 1; i < signal.length - 1; i++) {
        if (
            signal[i] > signal[i - 1] &&
            signal[i] > signal[i + 1] &&
            signal[i] > 0.3 // 閾値
        ) {
            peaks.push(i);
        }
    }

    if (peaks.length < 2) return null;

    const intervals = [];

    for (let i = 1; i < peaks.length; i++) {
        const t1 = time[peaks[i - 1]];
        const t2 = time[peaks[i]];
        intervals.push(t2 - t1);
    }

    if (intervals.length === 0) return null;

    const avgInterval =
        intervals.reduce((a, b) => a + b, 0) / intervals.length;

    const bpm = 60000 / avgInterval;

    // 妥当範囲チェック
    if (bpm < 40 || bpm > 180) return null;

    return bpm;
}
