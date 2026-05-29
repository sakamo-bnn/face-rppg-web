// face.js
// 役割:
// - 将来的には顔検出結果からROIを返す
// - 現段階では、画面中央にダミーの顔領域 / ROI を返す
// - main.js から detectFace(video) を呼び出して利用する

export function detectFace(video) {
    const width = video.videoWidth;
    const height = video.videoHeight;

    if (!width || !height) {
        return null;
    }

    // ===== ダミーの顔領域 =====
    // 画面中央付近に顔がある前提で仮の矩形を作る
    const faceWidth = width * 0.35;
    const faceHeight = height * 0.5;
    const faceX = (width - faceWidth) / 2;
    const faceY = (height - faceHeight) / 2;

    const faceBox = {
        x: faceX,
        y: faceY,
        width: faceWidth,
        height: faceHeight
    };

    // ===== ROI（額を想定）=====
    // 顔矩形の上側に細めの矩形を置く
    const roiWidth = faceWidth * 0.5;
    const roiHeight = faceHeight * 0.18;
    const roiX = faceX + (faceWidth - roiWidth) / 2;
    const roiY = faceY + faceHeight * 0.12;

    const roiBox = {
        x: roiX,
        y: roiY,
        width: roiWidth,
        height: roiHeight
    };

    return {
        faceBox,
        roiBox,
        landmarks: []
    };
}
