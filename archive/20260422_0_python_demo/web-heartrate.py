import os
import sys
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy import signal


CAMERA_INDEX = 0
TIMER_MS = 33

FACE_CONFIDENCE = 0.60
DETECTION_INTERVAL_SEC = 0.10
FACE_BOX_TTL_SEC = 1.0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

MIN_BPM = 45
MAX_BPM = 180
WINDOW_SEC = 10.0
BPM_UPDATE_INTERVAL = 1.0
RESAMPLE_FS = 30.0

CPU_COUNT = os.cpu_count() or 4
WORKER_COUNT = max(1, CPU_COUNT)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
PROTOTXT = os.path.join(ASSETS_DIR, "deploy.prototxt")
MODEL = os.path.join(ASSETS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(PROTOTXT):
    raise FileNotFoundError(f"deploy.prototxt が見つかりません: {PROTOTXT}")
if not os.path.exists(MODEL):
    raise FileNotFoundError(f"caffemodel が見つかりません: {MODEL}")


def safe_bandpass(sig, fs, low_hz=0.75, high_hz=3.0, order=3):
    if len(sig) < max(8, order * 3):
        return sig

    nyq = 0.5 * fs
    low = max(0.001, low_hz / nyq)
    high = min(0.99, high_hz / nyq)

    if low >= high:
        return sig

    b, a = signal.butter(order, [low, high], btype="bandpass")
    try:
        return signal.filtfilt(b, a, sig)
    except ValueError:
        return sig


def resample_signal(t, x, target_fs=30.0):
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if len(t) < 2 or len(x) < 2:
        return None, None

    t0, t1 = t[0], t[-1]
    if t1 <= t0:
        return None, None

    tu = np.arange(t0, t1, 1.0 / target_fs)
    if len(tu) < 8:
        return None, None

    xu = np.interp(tu, t, x)
    return tu, xu


def extract_chrom_signal(rgb_buffer):
    """
    rgb_buffer: shape (N, 3) [R, G, B]
    """
    X = np.asarray(rgb_buffer, dtype=np.float64)
    if len(X) < 2:
        return np.zeros(len(X), dtype=np.float64)

    mean_rgb = np.mean(X, axis=0)
    mean_rgb[mean_rgb == 0] = 1.0

    C = X / mean_rgb
    R = C[:, 0]
    G = C[:, 1]
    B = C[:, 2]

    Xsig = 3.0 * R - 2.0 * G
    Ysig = 1.5 * R + G - 1.5 * B

    std_y = np.std(Ysig)
    alpha = 0.0 if std_y < 1e-8 else np.std(Xsig) / std_y

    S = Xsig - alpha * Ysig
    S = S - np.mean(S)
    return S


def estimate_bpm_from_fft(sig, fs, min_bpm=45, max_bpm=180):
    n = len(sig)
    if n < 64 or fs <= 0:
        return None, 0.0

    x = np.asarray(sig, dtype=np.float64)
    x = x - np.mean(x)
    std = np.std(x)
    if std < 1e-6:
        return None, 0.0

    xw = x * np.hamming(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(xw))

    min_hz = min_bpm / 60.0
    max_hz = max_bpm / 60.0
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not np.any(mask):
        return None, 0.0

    freqs_sel = freqs[mask]
    spec_sel = spec[mask]

    peak_idx = np.argmax(spec_sel)
    peak_val = spec_sel[peak_idx]
    median_val = np.median(spec_sel) + 1e-8
    quality = float(peak_val / median_val)

    bpm = float(freqs_sel[peak_idx] * 60.0)
    return bpm, quality


def smooth_bpm(prev_bpm, new_bpm, alpha=0.08):
    if new_bpm is None:
        return prev_bpm
    if prev_bpm is None:
        return new_bpm
    return (1.0 - alpha) * prev_bpm + alpha * new_bpm


def build_net():
    return cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)


def detect_best_face_box(frame_bgr, net):
    img = cv2.resize(frame_bgr, (FRAME_WIDTH, FRAME_HEIGHT))
    h, w = img.shape[:2]

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


def roi_from_rel_box(img, face_box, rel):
    if face_box is None:
        return None, None

    h, w = img.shape[:2]
    x1, y1, x2, y2 = face_box
    fw = x2 - x1
    fh = y2 - y1

    rx1 = int(x1 + fw * rel[0])
    ry1 = int(y1 + fh * rel[1])
    rx2 = int(x1 + fw * rel[2])
    ry2 = int(y1 + fh * rel[3])

    rx1 = max(0, min(rx1, w - 1))
    ry1 = max(0, min(ry1, h - 1))
    rx2 = max(0, min(rx2, w - 1))
    ry2 = max(0, min(ry2, h - 1))

    if rx2 <= rx1 or ry2 <= ry1:
        return None, None

    roi = img[ry1:ry2, rx1:rx2]
    return roi, (rx1, ry1, rx2, ry2)


def get_multi_rois(img, face_box):
    forehead_rel = (0.28, 0.14, 0.72, 0.30)
    left_cheek_rel = (0.12, 0.45, 0.34, 0.68)
    right_cheek_rel = (0.66, 0.45, 0.88, 0.68)

    rois = []
    boxes = []

    for rel in [forehead_rel, left_cheek_rel, right_cheek_rel]:
        roi, box = roi_from_rel_box(img, face_box, rel)
        if roi is not None and roi.size > 0:
            rois.append(roi)
            boxes.append(box)

    return rois, boxes


def mean_rgb_from_rois(rois):
    if not rois:
        return None

    vals = []
    for roi in rois:
        roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
        b = float(np.mean(roi_blur[:, :, 0]))
        g = float(np.mean(roi_blur[:, :, 1]))
        r = float(np.mean(roi_blur[:, :, 2]))
        vals.append([r, g, b])

    vals = np.asarray(vals, dtype=np.float64)
    rgb = np.mean(vals, axis=0)
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


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


class FaceDetectionManager:
    def __init__(self, worker_count):
        self.executor = ThreadPoolExecutor(max_workers=worker_count)
        self.result_lock = threading.Lock()
        self.submit_lock = threading.Lock()
        self.worker_local = threading.local()

        self.in_flight = 0
        self.max_in_flight = worker_count

        self.latest_result = {
            "frame_ts": None,
            "box": None,
            "conf": None,
            "updated_at": 0.0,
        }

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


class PlotGraph:
    def __init__(self):
        self.app = QtWidgets.QApplication.instance()

        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle("RGB plot")
        self.plt = self.win.addPlot()
        self.plt.setLabel("left", "Intensity")
        self.plt.setLabel("bottom", "Samples")
        self.plt.showGrid(x=True, y=True)
        self.curve_r = self.plt.plot(pen=(255, 0, 0))
        self.curve_g = self.plt.plot(pen=(0, 255, 0))
        self.curve_b = self.plt.plot(pen=(0, 0, 255))

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

        self.rgb_history_len = 300
        self.data_r = deque(maxlen=self.rgb_history_len)
        self.data_g = deque(maxlen=self.rgb_history_len)
        self.data_b = deque(maxlen=self.rgb_history_len)

        self.ts = deque()
        self.rgb_signal = deque()

        self.last_bpm = None
        self.last_quality = 0.0
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

    def draw_face_window(self, img, box, conf, roi_boxes):
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

        for rb in roi_boxes or []:
            rx1, ry1, rx2, ry2 = rb
            cv2.rectangle(draw, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)

        cv2.imshow("Face Detection", draw)

    def get_rgb_sample_nonblocking(self):
        frame, frame_ts = self.camera.get_latest()
        if frame is None:
            return False, (np.nan, np.nan, np.nan), "camera not ready", None

        img = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        now = time.time()
        if now - self.last_detection_submit_time >= DETECTION_INTERVAL_SEC:
            self.detector.submit_if_possible(frame, frame_ts)
            self.last_detection_submit_time = now

        det = self.detector.get_latest_box()
        if det["box"] is not None:
            self.last_valid_face_box = det["box"]
            self.last_valid_face_conf = det["conf"]
            self.last_face_update_time = det["updated_at"]

        if (now - self.last_face_update_time) > FACE_BOX_TTL_SEC:
            self.last_valid_face_box = None
            self.last_valid_face_conf = None

        rois, roi_boxes = get_multi_rois(img, self.last_valid_face_box)
        self.draw_face_window(img, self.last_valid_face_box, self.last_valid_face_conf, roi_boxes)

        if not rois:
            return False, (np.nan, np.nan, np.nan), "roi empty", img

        rgb = mean_rgb_from_rois(rois)
        if rgb is None:
            return False, (np.nan, np.nan, np.nan), "rgb mean failed", img

        return True, rgb, "ok", img

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
                self.rgb_signal.append((r, g, b))

                while len(self.ts) > 0 and (now - self.ts[0]) > WINDOW_SEC:
                    self.ts.popleft()
                    self.rgb_signal.popleft()

                self.curve_r.setData(np.array(self.data_r, dtype=np.float64))
                self.curve_g.setData(np.array(self.data_g, dtype=np.float64))
                self.curve_b.setData(np.array(self.data_b, dtype=np.float64))

                bpm = self.last_bpm
                quality = self.last_quality

                if len(self.ts) >= 64:
                    t = np.array(self.ts, dtype=np.float64)
                    rgb_seq = np.array(self.rgb_signal, dtype=np.float64)

                    chrom_sig = extract_chrom_signal(rgb_seq)
                    tu, xu = resample_signal(t, chrom_sig, target_fs=RESAMPLE_FS)

                    if tu is not None and xu is not None and len(xu) >= 64:
                        xu = xu - np.mean(xu)
                        std = np.std(xu)
                        if std > 1e-6:
                            xu = xu / std

                        filtered = safe_bandpass(
                            xu,
                            RESAMPLE_FS,
                            low_hz=MIN_BPM / 60.0,
                            high_hz=MAX_BPM / 60.0,
                            order=3,
                        )

                        if len(filtered) >= 9:
                            filtered_disp = signal.savgol_filter(filtered, 9, 2)
                        else:
                            filtered_disp = filtered

                        if time.time() - self.last_bpm_update_time >= BPM_UPDATE_INTERVAL:
                            bpm_est, quality_est = estimate_bpm_from_fft(
                                filtered,
                                RESAMPLE_FS,
                                min_bpm=MIN_BPM,
                                max_bpm=MAX_BPM,
                            )

                            if bpm_est is not None and quality_est >= 2.2:
                                bpm = smooth_bpm(self.last_bpm, bpm_est, alpha=0.08)
                                self.last_bpm = bpm
                                self.last_quality = quality_est
                                quality = quality_est

                            self.last_bpm_update_time = time.time()

                        if len(filtered_disp) >= 7:
                            peak_idx, _ = signal.find_peaks(
                                filtered_disp,
                                distance=max(1, int(RESAMPLE_FS * 0.35)),
                                prominence=0.12,
                            )
                            if len(peak_idx) > 0:
                                self.curve_peak.setData(peak_idx, filtered_disp[peak_idx])
                            else:
                                self.curve_peak.setData([], [])
                        else:
                            self.curve_peak.setData([], [])

                        self.curve_sig_raw.setData(xu)
                        self.curve_sig.setData(filtered_disp)
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
                q_text = f"{quality:.2f}" if quality is not None else "--"

                print(
                    f"R: {r:.1f}, G: {g:.1f}, B: {b:.1f}, BPM: {bpm_text}, Q: {q_text}, FPS: {fps_text}"
                )
                self.plt2.setTitle(
                    f"BPM: {bpm_text} / Q: {q_text} / FPS: {fps_text} / workers: {WORKER_COUNT}"
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