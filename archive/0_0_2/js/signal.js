const MAX_BUFFER_SIZE = 300;

let signalBuffer = [];
let timeBuffer = [];

export function resetSignal() {
    signalBuffer = [];
    timeBuffer = [];
}

export function updateSignal(video, face) {
    if (!face || !face.roiBox) {
        return { signal: [], bpm: null };
    }

    const roi = face.roiBox;
    const g = getAverageG(video, roi);
    const now = performance.now();

    signalBuffer.push(g);
    timeBuffer.push(now);

    if (signalBuffer.length > MAX_BUFFER_SIZE) {
        signalBuffer.shift();
        timeBuffer.shift();
    }

    const normalized = normalize(signalBuffer);
    const bpm = estimateBPM(normalized, timeBuffer);

    return {
        signal: normalized,
        bpm
    };
}

function getAverageG(video, roi) {
    const canvas = getTempCanvas(video);
    const ctx = canvas.getContext("2d");

    const x = Math.max(0, Math.floor(roi.x));
    const y = Math.max(0, Math.floor(roi.y));
    const w = Math.floor(roi.width);
    const h = Math.floor(roi.height);

    if (w <= 0 || h <= 0) return 0;
    if (x + w > canvas.width || y + h > canvas.height) return 0;

    const imageData = ctx.getImageData(x, y, w, h);
    const data = imageData.data;

    let sumG = 0;
    let count = 0;

    for (let i = 0; i < data.length; i += 4) {
        sumG += data[i + 1];
        count++;
    }

    return count > 0 ? sumG / count : 0;
}

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

function normalize(arr) {
    if (arr.length === 0) return [];

    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const centered = arr.map((v) => v - mean);
    const max = Math.max(...centered.map((v) => Math.abs(v))) || 1;

    return centered.map((v) => v / max);
}

function estimateBPM(signal, time) {
    if (signal.length < 30) return null;

    const peaks = [];

    for (let i = 1; i < signal.length - 1; i++) {
        if (
            signal[i] > signal[i - 1] &&
            signal[i] > signal[i + 1] &&
            signal[i] > 0.3
        ) {
            peaks.push(i);
        }
    }

    if (peaks.length < 2) return null;

    const intervals = [];

    for (let i = 1; i < peaks.length; i++) {
        intervals.push(time[peaks[i]] - time[peaks[i - 1]]);
    }

    if (intervals.length === 0) return null;

    const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const bpm = 60000 / avgInterval;

    if (bpm < 40 || bpm > 180) return null;

    return bpm;
}