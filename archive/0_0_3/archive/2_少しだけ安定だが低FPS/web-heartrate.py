import os
import sys
import time
from collections import deque

import cv2
import imutils
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy import signal

# =========================
# 設定
# =========================
CAMERA_INDEX = 0

# 表示更新周期 [ms]
TIMER_MS = 33  # 約30fps相当

# 顔検出のしきい値
FACE_CONFIDENCE = 0.6

# BPM探索範囲 [bpm]
MIN_BPM = 45
MAX_BPM = 180

# 信号窓の長さ [秒]
WINDOW_SEC = 10.0

# BPMの更新間隔 [秒]
BPM_UPDATE_INTERVAL = 1.0

# 額ROIの比率
# 顔bboxに対して、額っぽい領域を切り出す
FOREHEAD_X1 = 0.25
FOREHEAD_X2 = 0.75
FOREHEAD_Y1 = 0.12
FOREHEAD_Y2 = 0.32

# =========================
# カメラ
# =========================
capture = cv2.VideoCapture(CAMERA_INDEX)

# WindowsでFPSが安定しやすいことがある
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv2.CAP_PROP_FPS, 30)

# =========================
# 顔検出モデル
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
prototxt = os.path.join(ASSETS_DIR, "deploy.prototxt")
model = os.path.join(ASSETS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(prototxt):
    raise FileNotFoundError(f"deploy.prototxt が見つかりません: {prototxt}")
if not os.path.exists(model):
    raise FileNotFoundError(f"caffemodel が見つかりません: {model}")

net = cv2.dnn.readNetFromCaffe(prototxt, model)


def safe_bandpass(sig, fs, low_hz=0.75, high_hz=3.0, order=3):
    """
    心拍相当帯域だけ通す
    0.75Hz = 45bpm
    3.0Hz  = 180bpm
    """
    if len(sig) < max(8, order * 3):
        return sig

    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq

    if high >= 1.0:
        high = 0.99
    if low <= 0.0:
        low = 0.001
    if low >= high:
        return sig

    b, a = signal.butter(order, [low, high], btype="bandpass")
    try:
        return signal.filtfilt(b, a, sig)
    except ValueError:
        return sig


def estimate_bpm_from_fft(sig, fs, min_bpm=45, max_bpm=180):
    """
    FFTピークからBPM推定
    """
    n = len(sig)
    if n < 32 or fs <= 0:
        return None

    x = np.asarray(sig, dtype=np.float64)
    x = x - np.mean(x)

    if np.std(x) < 1e-6:
        return None

    window = np.hamming(n)
    xw = x * window

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(xw))

    min_hz = min_bpm / 60.0
    max_hz = max_bpm / 60.0

    mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not np.any(mask):
        return None

    freqs_sel = freqs[mask]
    spec_sel = spec[mask]

    peak_idx = np.argmax(spec_sel)
    peak_hz = freqs_sel[peak_idx]
    bpm = peak_hz * 60.0
    return float(bpm)


def detect_face_and_get_roi(frame):
    """
    顔検出して額ROIを返す
    return:
      roi, display_frame, face_box, roi_box
    """
    display = frame.copy()
    img = imutils.resize(frame, width=640)
    (h, w) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()

    best_conf = -1.0
    best_box = None

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < FACE_CONFIDENCE:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        if conf > best_conf:
            best_conf = conf
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        cv2.imshow("Face Detection", img)
        return None, img, None, None

    x1, y1, x2, y2 = best_box
    fw = x2 - x1
    fh = y2 - y1

    # 額ROI
    rx1 = int(x1 + fw * FOREHEAD_X1)
    rx2 = int(x1 + fw * FOREHEAD_X2)
    ry1 = int(y1 + fh * FOREHEAD_Y1)
    ry2 = int(y1 + fh * FOREHEAD_Y2)

    rx1 = max(0, min(rx1, w - 1))
    ry1 = max(0, min(ry1, h - 1))
    rx2 = max(0, min(rx2, w - 1))
    ry2 = max(0, min(ry2, h - 1))

    roi = None
    if rx2 > rx1 and ry2 > ry1:
        roi = img[ry1:ry2, rx1:rx2]

    # 描画
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
    cv2.putText(
        img,
        f"face {best_conf:.2f}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Face Detection", img)
    return roi, img, best_box, (rx1, ry1, rx2, ry2)


def get_rgb_sample():
    """
    ROIの平均RGBを返す
    return:
      (ok, rgb_tuple, status_text)
    """
    ret, frame = capture.read()
    if not ret or frame is None:
        return False, (np.nan, np.nan, np.nan), "camera read failed"

    roi, _, face_box, roi_box = detect_face_and_get_roi(frame)

    if roi is None or roi.size == 0:
        return False, (np.nan, np.nan, np.nan), "roi empty"

    # ROIを少し平滑化
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)

    # OpenCVはBGR
    b = float(np.mean(roi_blur[:, :, 0]))
    g = float(np.mean(roi_blur[:, :, 1]))
    r = float(np.mean(roi_blur[:, :, 2]))

    return True, (r, g, b), "ok"


class PlotGraph:
    def __init__(self):
        self.app = QtWidgets.QApplication.instance()

        # ---- ウィンドウ1: RGB ----
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle("RGB plot")
        self.plt = self.win.addPlot()
        self.plt.setLabel("left", "Intensity")
        self.plt.setLabel("bottom", "Samples")
        self.plt.showGrid(x=True, y=True)
        self.curve_r = self.plt.plot(pen=(255, 0, 0), name="R")
        self.curve_g = self.plt.plot(pen=(0, 255, 0), name="G")
        self.curve_b = self.plt.plot(pen=(0, 0, 255), name="B")

        # ---- ウィンドウ2: rPPG ----
        self.win2 = pg.GraphicsLayoutWidget(show=True)
        self.win2.setWindowTitle("rPPG / BPM plot")
        self.plt2 = self.win2.addPlot()
        self.plt2.setLabel("left", "Normalized amplitude")
        self.plt2.setLabel("bottom", "Samples")
        self.plt2.showGrid(x=True, y=True)
        self.curve_sig_raw = self.plt2.plot(pen=(180, 180, 180))
        self.curve_sig = self.plt2.plot(pen=(0, 255, 255))
        self.curve_peak = self.plt2.plot(
            pen=None, symbol="o", symbolPen=None, symbolSize=6, symbolBrush="r"
        )

        # ---- バッファ ----
        self.rgb_history_len = 300
        self.data_r = deque(maxlen=self.rgb_history_len)
        self.data_g = deque(maxlen=self.rgb_history_len)
        self.data_b = deque(maxlen=self.rgb_history_len)

        self.ts = deque()
        self.g_signal = deque()

        self.last_bpm = None
        self.last_bpm_update_time = 0.0
        self.last_update_time = None
        self.fps_smooth = None

        # ---- タイマー ----
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(TIMER_MS)

    def update_fps(self):
        now = time.time()
        if self.last_update_time is None:
            self.last_update_time = now
            return None

        dt = now - self.last_update_time
        self.last_update_time = now
        if dt <= 0:
            return self.fps_smooth

        fps = 1.0 / dt
        if self.fps_smooth is None:
            self.fps_smooth = fps
        else:
            self.fps_smooth = 0.9 * self.fps_smooth + 0.1 * fps

        return self.fps_smooth

    def update(self):
        try:
            ok, (r, g, b), status = get_rgb_sample()
            fps = self.update_fps()

            if ok:
                self.data_r.append(r)
                self.data_g.append(g)
                self.data_b.append(b)

                now = time.time()
                self.ts.append(now)
                self.g_signal.append(g)

                # 古いデータ削除
                while len(self.ts) > 0 and (now - self.ts[0]) > WINDOW_SEC:
                    self.ts.popleft()
                    self.g_signal.popleft()

                # ---- RGB plot ----
                self.curve_r.setData(np.array(self.data_r))
                self.curve_g.setData(np.array(self.data_g))
                self.curve_b.setData(np.array(self.data_b))

                bpm = self.last_bpm
                filtered = None
                peak_idx = np.array([], dtype=int)

                if len(self.ts) >= 32:
                    t = np.array(self.ts, dtype=np.float64)
                    x = np.array(self.g_signal, dtype=np.float64)

                    duration = t[-1] - t[0]
                    if duration > 1.5:
                        fs = (len(t) - 1) / duration

                        # DC除去 + 標準化
                        x_norm = x - np.mean(x)
                        std = np.std(x_norm)
                        if std > 1e-6:
                            x_norm = x_norm / std

                        # バンドパス
                        filtered = safe_bandpass(
                            x_norm,
                            fs,
                            low_hz=MIN_BPM / 60.0,
                            high_hz=MAX_BPM / 60.0,
                            order=3,
                        )

                        # 表示用にさらに軽く平滑化
                        if len(filtered) >= 5:
                            filtered_disp = signal.savgol_filter(
                                filtered,
                                window_length=5 if len(filtered) >= 5 else len(filtered) // 2 * 2 + 1,
                                polyorder=2
                            )
                        else:
                            filtered_disp = filtered

                        # BPM更新は毎秒程度
                        if time.time() - self.last_bpm_update_time >= BPM_UPDATE_INTERVAL:
                            bpm_est = estimate_bpm_from_fft(
                                filtered,
                                fs,
                                min_bpm=MIN_BPM,
                                max_bpm=MAX_BPM,
                            )
                            if bpm_est is not None:
                                # いきなり跳ねないように平滑化
                                if self.last_bpm is None:
                                    self.last_bpm = bpm_est
                                else:
                                    self.last_bpm = 0.8 * self.last_bpm + 0.2 * bpm_est
                                bpm = self.last_bpm
                            self.last_bpm_update_time = time.time()

                        # ピーク描画
                        if len(filtered_disp) >= 7:
                            peak_idx, _ = signal.find_peaks(
                                filtered_disp,
                                distance=max(1, int(fs * 0.35)),
                                prominence=0.15
                            )

                        self.curve_sig_raw.setData(x_norm)
                        self.curve_sig.setData(filtered_disp)

                        if len(peak_idx) > 0:
                            self.curve_peak.setData(peak_idx, filtered_disp[peak_idx])
                        else:
                            self.curve_peak.setData([], [])
                    else:
                        self.curve_sig_raw.setData([])
                        self.curve_sig.setData([])
                        self.curve_peak.setData([], [])
                else:
                    self.curve_sig_raw.setData([])
                    self.curve_sig.setData([])
                    self.curve_peak.setData([], [])

                # コンソール表示
                bpm_text = f"{bpm:.1f}" if bpm is not None else "--"
                fps_text = f"{fps:.1f}" if fps is not None else "--"
                print(
                    f"R: {r:.1f}, G: {g:.1f}, B: {b:.1f}, BPM: {bpm_text}, FPS: {fps_text}"
                )

                title = f"BPM: {bpm_text} / FPS: {fps_text}"
                self.plt2.setTitle(title)

            else:
                fps_text = f"{fps:.1f}" if fps is not None else "--"
                print(f"status: {status}, BPM: --, FPS: {fps_text}")
                self.plt2.setTitle(f"BPM: -- / FPS: {fps_text} / {status}")

        except Exception as e:
            print(f"update error: {repr(e)}")

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            self.close_all()

    def close_all(self):
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            capture.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        QtWidgets.QApplication.quit()


if __name__ == "__main__":
    graphWin = PlotGraph()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtWidgets.QApplication.instance().exec_()