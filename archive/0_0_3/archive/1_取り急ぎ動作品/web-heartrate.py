import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np
import sys
import cv2
from scipy import signal
from scipy import fftpack

#import face_recognition
import imutils

capture = cv2.VideoCapture(0)

# モデルを読み込む
# prototxt = '/Users/yuma/sample/deploy.prototxt'
# model = '/Users/yuma/sample/res10_300x300_ssd_iter_140000.caffemodel'

# prototxt = r'archive\20260422_0_引継ぎ\assets\deploy.prototxt'
# model = r'archive\20260422_0_引継ぎ\assets\res10_300x300_ssd_iter_140000.caffemodel'

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
prototxt = os.path.join(BASE_DIR, "assets", "deploy.prototxt")
model = os.path.join(BASE_DIR, "assets", "res10_300x300_ssd_iter_140000.caffemodel")

net = cv2.dnn.readNetFromCaffe(prototxt, model)
    
# def get_gbr():  
#     try:
#         ret, frame = capture.read()
#         # 明るさとコントラストを調整
#         frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
#         img = imutils.resize(frame)
#         (h, w) = img.shape[:2]
#         blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#         # 物体検出器にblobを適用する
#         net.setInput(blob)
#         detections = net.forward()
        
#         for i in range(0, detections.shape[2]):
#             # ネットワークが出力したconfidenceの値を抽出する
#             confidence = detections[0, 0, i, 2]
#             # confidenceの値が0.5以上の領域のみを検出結果として描画する
#             if confidence > 0.5:
#                 # 対象領域のバウンディングボックスの座標を計算する
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")
#                 midX = startX + (endX-startX)/2
#                 midY = startY + (endY-startY)/2
#                 startX = int(midX)
#                 startY = int(midY)
#                 endX = int(midX + 100)
#                 endY = int(midY - 100)
#                 # バウンディングボックスとconfidenceの値を描画する
#                 text = "{:.2f}%".format(confidence * 100)
#                 y = startY - 10 if startY - 10 > 10 else startY + 10
#                 cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
#                 cv2.putText(img, text, (startX, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

#         cv2.imshow("Face Detection", img)
        
#         frame_color = frame[1][0]
#         frame_area = frame[startX:endX,startY:endY]
        
#         # メディアンフィルタによる平滑化
#         images_green_median = [cv2.medianBlur(src=frame_area, ksize=5)]
        
#         rsum,gsum,bsum,i = 0,0,0,0
#         for raster in frame_area:
#             for px in raster:
#                 i += 1
#                 rsum += px[2]
#                 gsum += px[0]
#                 bsum += px[1]
                
                
#         # 平均値
#         ravg = rsum / 100
#         gavg = gsum / 100
#         bavg = bsum / 100
        
#         ave = [gavg, bavg, ravg]
        
#         print(f"R: {ravg}, G: {gavg}, B: {bavg}")
        
#     # 画素が取得できなかった時の例外処理
#     except:
#         print('err')
#         return [123,123,123]
    
#     return ave  

def get_gbr():
    try:
        ret, frame = capture.read()
        if not ret or frame is None:
            print("camera read failed")
            return [123, 123, 123]

        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
        img = imutils.resize(frame)
        (h, w) = img.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()

        roi = None

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                midX = int((x1 + x2) / 2)
                midY = int((y1 + y2) / 2)

                # 額あたりを想定したROI
                x1 = max(midX - 50, 0)
                x2 = min(midX + 50, w)
                y1 = max(midY - 120, 0)
                y2 = min(midY - 20, h)

                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                break

        cv2.imshow("Face Detection", img)

        if roi is None or roi.size == 0:
            print("roi empty")
            return [123, 123, 123]

        r = np.mean(roi[:, :, 2])
        g = np.mean(roi[:, :, 1])
        b = np.mean(roi[:, :, 0])

        print(f"R: {r:.1f}, G: {g:.1f}, B: {b:.1f}")
        return [g, b, r]

    except Exception as e:
        print("get_gbr error:", repr(e))
        return [123, 123, 123]
    
class PlotGraph:
    def __init__(self):
        # UIを設定
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle('RGB plot')
        self.plt = self.win.addPlot()
        self.plt.setYRange(0, 255)
        self.curve_r = self.plt.plot(pen=(255, 0, 0))
        self.curve_g = self.plt.plot(pen=(0, 255, 0))
        self.curve_b = self.plt.plot(pen=(0, 0, 255))
        
        self.win2 = pg.GraphicsLayoutWidget(show=True)
        self.win2.setWindowTitle('Ro plot')
        self.plt2 = self.win2.addPlot()
        self.plt2.setYRange(0, 255)
        
        self.curve_g_smg = self.plt2.plot(pen=(0, 255, 255))
        self.curve_g_peak = self.plt2.plot(pen=(0, 255, 255))
        
        # データを更新する関数を呼び出す時間を設定
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)
        
        self.data_r = np.zeros((100))
        self.data_g = np.zeros((100))
        self.data_b = np.zeros((100))
        self.data = []
        
        # 追加
        self.time_data = []
        
    # def update(self):
    #     self.data_r = np.delete(self.data_r, 0)
    #     self.data_g = np.delete(self.data_g, 0)
    #     self.data_b = np.delete(self.data_b, 0)
    #     gbr = get_gbr()
        
    #     self.data_r = np.append(self.data_r, gbr[2])
    #     self.data_g = np.append(self.data_g, gbr[0])
    #     self.data_b = np.append(self.data_b, gbr[1])
    #     self.data.append(gbr[0])
        
    #     self.curve_r.setData(self.data_r)
    #     self.curve_g.setData(self.data_g)
    #     self.curve_b.setData(self.data_b)
        
    #     window = 5 # 移動平均の範囲
    #     w = np.ones(window)/window
    #     x = np.convolve(self.data_g, w, mode='same')
    #     self.curve_g_smg.setData(x)
        
    #     N = 100
    #     threshold = 0.6 # 振幅の閾値
        
    #     x = np.fft.fft(self.data_g)
    #     x_abs = np.abs(x)
    #     x_abs = x_abs / N * 2
    #     x[x_abs < threshold] = 0

    #     x = np.fft.ifft(x)
    #     x = x.real # 複素数から実数部だけ取り出す
    #     self.curve_g_smg.setData(x)

    #     #ピーク値のインデックスを赤色で描画
    #     maxid = signal.argrelmax(x, order=3) #最大値
    #     minid = signal.argrelmin(x, order=1) #最小値
    #     self.curve_g_peak.setData(maxid, x[maxid], pen=None, symbol='o', symbolPen=None, symbolSize=4, symbolBrush=('r'))

    def update(self):
        self.data_r = np.delete(self.data_r, 0)
        self.data_g = np.delete(self.data_g, 0)
        self.data_b = np.delete(self.data_b, 0)

        gbr = get_gbr()

        self.data_r = np.append(self.data_r, gbr[2])
        self.data_g = np.append(self.data_g, gbr[0])
        self.data_b = np.append(self.data_b, gbr[1])

        now = QtCore.QDateTime.currentDateTime().toMSecsSinceEpoch() / 1000.0
        self.time_data.append(now)
        if len(self.time_data) > 100:
            self.time_data.pop(0)

        self.curve_r.setData(self.data_r)
        self.curve_g.setData(self.data_g)
        self.curve_b.setData(self.data_b)

        window = 5
        w = np.ones(window) / window
        x = np.convolve(self.data_g, w, mode='same')

        N = len(x)
        threshold = 0.6

        xf = np.fft.fft(x)
        x_abs = np.abs(xf)
        x_abs = x_abs / N * 2
        xf[x_abs < threshold] = 0

        x = np.fft.ifft(xf)
        x = x.real
        self.curve_g_smg.setData(x)

        maxid = signal.argrelmax(x, order=3)[0]
        self.curve_g_peak.setData(
            maxid, x[maxid],
            pen=None, symbol='o', symbolPen=None,
            symbolSize=4, symbolBrush=('r')
        )

        bpm = 0.0
        if len(maxid) >= 2 and len(self.time_data) == len(self.data_g):
            peak_times = [self.time_data[i] for i in maxid if i < len(self.time_data)]

            if len(peak_times) >= 2:
                intervals = np.diff(peak_times)
                mean_interval = np.mean(intervals)

                if mean_interval > 0:
                    bpm = 60.0 / mean_interval

        print(f"R: {gbr[2]:.1f}, G: {gbr[0]:.1f}, B: {gbr[1]:.1f}, BPM: {bpm:.1f}")
        
if __name__ == "__main__":
    graphWin = PlotGraph()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()
