export function drawOverlay(canvas, face) {
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    if (!face) {
        return;
    }

    if (face.faceBox) {
        drawBox(ctx, face.faceBox, {
            strokeStyle: "#00bcd4",
            lineWidth: 2,
            label: "Face"
        });
    }

    if (face.roiBox) {
        drawBox(ctx, face.roiBox, {
            strokeStyle: "#4caf50",
            lineWidth: 2,
            label: "ROI"
        });
    }

    if (Array.isArray(face.landmarks) && face.landmarks.length > 0) {
        drawLandmarks(ctx, face.landmarks);
    }
}

export function drawWaveform(canvas, signal) {
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    if (!signal || signal.length < 2) {
        drawWaveformMessage(ctx, width, height, "signal: no data");
        return;
    }

    ctx.strokeStyle = "#4caf50";
    ctx.lineWidth = 2;
    ctx.beginPath();

    const len = signal.length;

    for (let i = 0; i < len; i++) {
        const x = (i / (len - 1)) * width;
        const y = height / 2 - signal[i] * (height * 0.4);

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }

    ctx.stroke();
}

function drawBox(ctx, box, options = {}) {
    const {
        strokeStyle = "#fff",
        lineWidth = 2,
        label = ""
    } = options;

    ctx.save();

    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(box.x, box.y, box.width, box.height);

    if (label) {
        const labelPaddingX = 6;
        const labelHeight = 20;

        ctx.font = "14px Arial";
        const textWidth = ctx.measureText(label).width;
        const bgWidth = textWidth + labelPaddingX * 2;
        const bgX = box.x;
        const bgY = Math.max(0, box.y - labelHeight);

        ctx.fillStyle = strokeStyle;
        ctx.fillRect(bgX, bgY, bgWidth, labelHeight);

        ctx.fillStyle = "#000";
        ctx.textBaseline = "middle";
        ctx.fillText(label, bgX + labelPaddingX, bgY + labelHeight / 2);
    }

    ctx.restore();
}

function drawLandmarks(ctx, landmarks) {
    ctx.save();
    ctx.fillStyle = "#ffeb3b";

    for (const point of landmarks) {
        if (typeof point.x !== "number" || typeof point.y !== "number") {
            continue;
        }

        ctx.beginPath();
        ctx.arc(point.x, point.y, 2, 0, Math.PI * 2);
        ctx.fill();
    }

    ctx.restore();
}

function drawWaveformMessage(ctx, width, height, text) {
    ctx.save();
    ctx.fillStyle = "#888";
    ctx.font = "14px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, width / 2, height / 2);
    ctx.restore();
}