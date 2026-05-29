import {
    FaceLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

let faceLandmarker = null;
let initialized = false;
let lastVideoTime = -1;
let lastFaceResult = null;

export async function initFaceDetector() {
    if (initialized) return;

    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        },
        runningMode: "VIDEO",
        numFaces: 1
    });

    initialized = true;
}

export async function detectFace(video) {
    if (!faceLandmarker) {
        return null;
    }

    if (!video || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
        return null;
    }

    const currentTime = video.currentTime;

    if (currentTime === lastVideoTime) {
        return lastFaceResult;
    }

    lastVideoTime = currentTime;

    const result = faceLandmarker.detectForVideo(video, performance.now());

    if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
        lastFaceResult = null;
        return null;
    }

    const landmarks = result.faceLandmarks[0];
    const points = landmarks.map((p) => ({
        x: p.x * video.videoWidth,
        y: p.y * video.videoHeight
    }));

    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);

    const minX = Math.max(0, Math.min(...xs));
    const maxX = Math.min(video.videoWidth, Math.max(...xs));
    const minY = Math.max(0, Math.min(...ys));
    const maxY = Math.min(video.videoHeight, Math.max(...ys));

    const faceBox = {
        x: minX,
        y: minY,
        width: Math.max(0, maxX - minX),
        height: Math.max(0, maxY - minY)
    };

    if (faceBox.width <= 0 || faceBox.height <= 0) {
        lastFaceResult = null;
        return null;
    }

    const roiWidth = faceBox.width * 0.35;
    const roiHeight = faceBox.height * 0.14;
    const roiX = faceBox.x + (faceBox.width - roiWidth) / 2;
    const roiY = faceBox.y + faceBox.height * 0.12;

    const roiBox = {
        x: Math.max(0, roiX),
        y: Math.max(0, roiY),
        width: Math.min(roiWidth, video.videoWidth - roiX),
        height: Math.min(roiHeight, video.videoHeight - roiY)
    };

    lastFaceResult = {
        faceBox,
        roiBox,
        landmarks: points
    };

    return lastFaceResult;
}