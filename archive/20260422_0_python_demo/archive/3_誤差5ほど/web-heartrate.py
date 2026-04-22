import os
import sys
import time
import threading
import queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor

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
TIMER_MS = 33  # UI更新周期
FACE_CONFIDENCE = 0.6
MIN_BPM = 45
MAX_BPM = 180
WINDOW_SEC = 10.0
BPM_UPDATE_INTERVAL = 1.0

FOREHEAD_X1 = 0.25
FOREHEAD_X2 = 0.75
FOREHEAD_Y1 = 0.12
FOREHEAD_Y2 = 0.32

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# 顔検出投入間隔
DETECTION_INTERVAL_SEC = 0.08  # 約12.5Hz
# 顔未更新時のbbox保持時間
FACE_BOX_TTL_SEC = 1.0

CPU_COUNT = os.cpu_count() or 4
WORKER_COUNT = max(1, CPU_COUNT)

# =========================
# モデルパス
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
PROTOTXT = os.path.join(ASSETS_DIR, "deploy.prototxt")
MODEL = os.path.join(ASSETS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(PROTOTXT):
    raise FileNotFoundError(f"deploy.prototxt が見つかりません: {PROTOTXT}")
if not os.path.exists(MODEL):
    raise FileNotFoundError(f"caffemodel が見つかりません: {MODEL}")


# =========================
# 信号処理
# =========================
def safe_bandpass(sig, fs, low_hz=0.75, high_hz=3.0, order=3):
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
    return float(peak_hz * 60.0)


# =========================
# 顔検出ワーカー用
# =========================
def build_net():
    return cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)


def detect_best_face_box(frame_bgr, net):
    """
    入力: BGR frame
    出力: (resized_frame, best_box, best_conf)
      best_box は resized_frame 座標系
    """
    img = imutils.resize(frame_bgr, width=FRAME_WIDTH)
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

    return img, best_box, best_conf


def forehead_roi_from_box(img, box):
    if box is None:
        return None, None

    (h, w) = img.shape[:2]
    x1, y1, x2, y2 = box
    fw = x2 - x1
    fh = y2 - y1

    rx1 = int(x1 + fw * FOREHEAD_X1)
    rx2 = int(x1 + fw * FOREHEAD_X2)
    ry1 = int(y1 + fh * FOREHEAD_Y1)
    ry2 = int(y1 + fh * FOREHEAD_Y2)

    rx1 = max(0, min(rx1, w - 1))
    ry1 = max(0, min(ry1, h - 1))
    rx2 = max(0, min(rx2, w - 1))
    ry2 = max(0, min(ry2, h - 1))

    if rx2 <= rx1 or ry2 <= ry1:
        return None, None

    roi = img[ry1:ry2, rx1:rx2]
    return roi, (rx1, ry1, rx2, ry2)


# =========================
# カメラキャプチャスレッド
# =========================
class CameraReader:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_ts = None
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_ts = time.time()

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None, None
            return self.latest_frame.copy(), self.latest_ts

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass


# =========================
# 顔検出マネージャ
# =========================
class FaceDetectionManager:
    def __init__(self, worker_count):
        self.worker_count = worker_count
        self.executor = ThreadPoolExecutor(max_workers=worker_count)
        self.result_lock = threading.Lock()

        self.latest_result = {
            "frame_ts": None,
            "box": None,
            "conf": None,
            "updated_at": 0.0,
        }

        self.submit_lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = worker_count
        self.worker_local = threading.local()

    def _get_thread_net(self):
        if not hasattr(self.worker_local, "net"):
            self.worker_local.net = build_net()
        return self.worker_local.net

    def _task(self, frame, frame_ts):
        net = self._get_thread_net()
        _, box, conf = detect_best_face_box(frame, net)
        return frame_ts, box, conf

    def _done_callback(self, fut):
        with self.submit_lock:
            self.in_flight = max(0, self.in_flight - 1)

        try:
            frame_ts, box, conf = fut.result()
        except Exception:
            return

        with self.result_lock:
            prev_ts = self.latest_result["frame_ts"]
            if prev_ts is None or frame_ts >= prev_ts:
                self.latest_result = {
                    "frame_ts": frame_ts,
                    "box": box,
                    "conf": conf,
                    "updated_at": time.time(),
                }

    def submit_if_possible(self, frame, frame_ts):
        with self.submit_lock:
            if self.in_flight >= self.max_in_flight:
                return False
            self.in_flight += 1

        fut = self.executor.submit(self._task, frame.copy(), frame_ts)
        fut.add_done_callback(self._done_callback)
        return True

    def get_latest_box(self):
        with self.result_lock:
            return dict(self.latest_result)

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


# =========================
# メインGUI
# =========================
class PlotGraph:
    def __init__(self):
        self.app = QtWidgets.QApplication.instance()

        # 表示
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle("RGB plot")
        self.plt = self.win.addPlot()
        self.plt.setLabel("left", "Intensity")
        self.plt.setLabel("bottom", "Samples")
        self.plt.showGrid(x=True, y=True)
        self.curve_r = self.plt.plot(pen=(255, 0, 0), name="R")
        self.curve_g = self.plt.plot(pen=(0, 255, 0), name="G")
        self.curve_b = self.plt.plot(pen=(0, 0, 255), name="B")

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

        # バッファ
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

        self.camera = CameraReader(CAMERA_INDEX)
        self.detector = FaceDetectionManager(WORKER_COUNT)
        self.camera.start()

        self.last_detection_submit_time = 0.0
        self.last_valid_face_box = None
        self.last_valid_face_conf = None
        self.last_face_update_time = 0.0

        self.last_frame_draw = None

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(TIMER_MS)

        print(f"CPU logical cores: {CPU_COUNT}, worker threads: {WORKER_COUNT}")

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

    def draw_face_window(self, img, box, conf, roi_box):
        draw = img.copy()

        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if conf is not None:
                cv2.putText(
                    draw,
                    f"face {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        if roi_box is not None:
            rx1, ry1, rx2, ry2 = roi_box
            cv2.rectangle(draw, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)

        cv2.imshow("Face Detection", draw)

    def get_rgb_sample_nonblocking(self):
        frame, frame_ts = self.camera.get_latest()
        if frame is None:
            return False, (np.nan, np.nan, np.nan), "camera not ready", None

        img = imutils.resize(frame, width=FRAME_WIDTH)

        # 新規検出投入
        now = time.time()
        if now - self.last_detection_submit_time >= DETECTION_INTERVAL_SEC:
            self.detector.submit_if_possible(frame, frame_ts)
            self.last_detection_submit_time = now

        # 最新検出結果を取得
        det = self.detector.get_latest_box()
        if det["box"] is not None:
            self.last_valid_face_box = det["box"]
            self.last_valid_face_conf = det["conf"]
            self.last_face_update_time = det["updated_at"]

        # 古いbboxは無効化
        if (now - self.last_face_update_time) > FACE_BOX_TTL_SEC:
            self.last_valid_face_box = None
            self.last_valid_face_conf = None

        roi, roi_box = forehead_roi_from_box(img, self.last_valid_face_box)
        self.draw_face_window(img, self.last_valid_face_box, self.last_valid_face_conf, roi_box)

        if roi is None or roi.size == 0:
            return False, (np.nan, np.nan, np.nan), "roi empty", img

        roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)

        b = float(np.mean(roi_blur[:, :, 0]))
        g = float(np.mean(roi_blur[:, :, 1]))
        r = float(np.mean(roi_blur[:, :, 2]))
        return True, (r, g, b), "ok", img

    def update(self):
        try:
            ok, (r, g, b), status, img = self.get_rgb_sample_nonblocking()
            fps = self.update_fps()

            if ok:
                self.data_r.append(r)
                self.data_g.append(g)
                self.data_b.append(b)

                now = time.time()
                self.ts.append(now)
                self.g_signal.append(g)

                while len(self.ts) > 0 and (now - self.ts[0]) > WINDOW_SEC:
                    self.ts.popleft()
                    self.g_signal.popleft()

                self.curve_r.setData(np.array(self.data_r))
                self.curve_g.setData(np.array(self.data_g))
                self.curve_b.setData(np.array(self.data_b))

                bpm = self.last_bpm
                peak_idx = np.array([], dtype=int)

                if len(self.ts) >= 32:
                    t = np.array(self.ts, dtype=np.float64)
                    x = np.array(self.g_signal, dtype=np.float64)

                    duration = t[-1] - t[0]
                    if duration > 1.5:
                        fs = (len(t) - 1) / duration

                        x_norm = x - np.mean(x)
                        std = np.std(x_norm)
                        if std > 1e-6:
                            x_norm = x_norm / std

                        filtered = safe_bandpass(
                            x_norm,
                            fs,
                            low_hz=MIN_BPM / 60.0,
                            high_hz=MAX_BPM / 60.0,
                            order=3,
                        )

                        if len(filtered) >= 5:
                            filtered_disp = signal.savgol_filter(
                                filtered,
                                window_length=5,
                                polyorder=2,
                            )
                        else:
                            filtered_disp = filtered

                        if time.time() - self.last_bpm_update_time >= BPM_UPDATE_INTERVAL:
                            bpm_est = estimate_bpm_from_fft(
                                filtered,
                                fs,
                                min_bpm=MIN_BPM,
                                max_bpm=MAX_BPM,
                            )
                            if bpm_est is not None:
                                if self.last_bpm is None:
                                    self.last_bpm = bpm_est
                                else:
                                    self.last_bpm = 0.8 * self.last_bpm + 0.2 * bpm_est
                                bpm = self.last_bpm
                            self.last_bpm_update_time = time.time()

                        if len(filtered_disp) >= 7:
                            peak_idx, _ = signal.find_peaks(
                                filtered_disp,
                                distance=max(1, int(fs * 0.35)),
                                prominence=0.15,
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

                bpm_text = f"{bpm:.1f}" if bpm is not None else "--"
                fps_text = f"{fps:.1f}" if fps is not None else "--"
                print(f"R: {r:.1f}, G: {g:.1f}, B: {b:.1f}, BPM: {bpm_text}, FPS: {fps_text}")

                self.plt2.setTitle(
                    f"BPM: {bpm_text} / FPS: {fps_text} / workers: {WORKER_COUNT}"
                )
            else:
                fps_text = f"{fps:.1f}" if fps is not None else "--"
                print(f"status: {status}, BPM: --, FPS: {fps_text}")
                self.plt2.setTitle(
                    f"BPM: -- / FPS: {fps_text} / workers: {WORKER_COUNT} / {status}"
                )

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
            self.camera.stop()
        except Exception:
            pass
        try:
            self.detector.shutdown()
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