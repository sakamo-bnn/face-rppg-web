
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
cd archive\20260422_1_from_python_demo
python -m http.server 8000

# Web Heart Rate Monitor

## 概要
添付された `web-heartrate.py` の構成を、ブラウザで動作する Web アプリへ置き換えた実装です。

- カメラ取得: `getUserMedia`
- 顔検出: MediaPipe Face Detector
- グラフ描画: Chart.js
- 信号処理: 純粋な JavaScript 実装（CHROM, 補間, FFT ベース BPM 推定）

## ファイル構成
- `index.html` UI 本体
- `styles.css` 見た目
- `app.js` カメラ制御・顔検出・ROI 抽出・rPPG 推定・グラフ更新

## 起動方法
ローカルでそのまま `index.html` を開くより、簡易 HTTP サーバを使う方が安定します。

### Python を使う場合
```bash
cd web_heartrate_app
python -m http.server 8000
```

その後ブラウザで `http://localhost:8000` を開いてください。

## 注意
- 元の Python 実装は OpenCV DNN の Caffe 顔検出器、SciPy の Butterworth + filtfilt、PyQtGraph を使っていました。
- この Web 版では、顔検出は MediaPipe、フィルタは JavaScript 側の軽量近似実装へ置き換えています。
- そのため BPM の数値は元アプリと完全一致ではありませんが、構成と流れは対応しています。
