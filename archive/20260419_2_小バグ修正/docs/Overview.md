# 顔の脈波検出アプリ開発ロードマップ

## 目的

- 顔のROIからrPPGを取得し、脈波らしい波形とBPMを表示する
- まずはPC上でWeb PoCを成立させる
- 次にTablet上でWeb版をそのまま検証する
- 必要な場合のみ、Androidネイティブアプリへ移行する
- Androidネイティブ化の際も、できるだけ単純な技術構成を保つ

---

## 全体方針

開発は次の3段階で進める。

1. PCのWebカメラでWeb PoCを作る
2. TabletでWeb版をそのまま動かして確認する
3. 必要ならAndroidネイティブアプリへ移行する

この順序にすることで、

- アルゴリズム由来の問題
- Tablet実機やブラウザ由来の問題

を切り分けやすくなる。

---

## Step 1: PCのWebカメラでWeb PoC

### 目的

- 顔が取れる
- ROIが安定する
- BPMがだいたい出る

### 推奨技術構成

#### 言語
- JavaScript

#### 画面
- 単純なHTML
- CSS
- Canvas

#### カメラ取得
- getUserMedia

#### 顔検出・顔ランドマーク
- MediaPipe Face Landmarker

#### 信号処理
- JavaScriptで自前実装
- 必要に応じて軽量FFTライブラリを追加

### 最小ファイル構成

- index.html
- style.css
- main.js
- face.js
- signal.js
- draw.js

### 各ファイルの役割

#### index.html
- video要素を置く
- canvas要素を置く
- BPM表示領域を置く
- 状態表示領域を置く

#### main.js
- カメラ起動
- MediaPipe初期化
- 全体ループ制御
- BPM更新

#### face.js
- 顔ランドマーク推定
- 額または頬のROI決定
- ROIの安定化

#### signal.js
- ROI内平均RGB取得
- 時系列バッファ管理
- 正規化
- detrend
- バンドパス
- BPM推定

#### draw.js
- 顔枠表示
- ROI表示
- 波形表示
- デバッグ表示

### 実装ステップ

#### 1-1. カメラ映像を表示する
- getUserMediaでWebカメラを起動する
- videoに映像を表示する
- 正常に映ることを確認する

#### 1-2. 顔を検出する
- MediaPipe Face Landmarkerを導入する
- 顔ランドマークを取得する
- 顔位置が追従することを確認する

#### 1-3. ROIを決める
- 額または両頬を候補とする
- 顔ランドマークからROI矩形を計算する
- ROIが顔の動きに追従することを確認する

#### 1-4. ROIの平均色を時系列化する
- ROI内ピクセルの平均RGB値を各フレームで計算する
- 一定長のリングバッファに保存する
- まずはG成分を中心に扱う

#### 1-5. 基本的な信号処理を入れる
- 平均除去
- 正規化
- detrend
- バンドパス
- 平滑化

#### 1-6. BPMを推定する
- ピーク間隔からBPMを求める
- または周波数成分からBPMを求める
- 数値がだいたい妥当か確認する

#### 1-7. 可視化を入れる
- ROIをCanvasに描画する
- 波形をCanvasに描画する
- BPMを数値表示する
- 簡単な品質指標を表示する

### Step 1の完了条件

- PCのWebカメラで顔検出が安定する
- ROIが大きく崩れない
- 波形らしい変動が見える
- BPMが大きく外れずに出る

---

## Step 2: TabletでWeb版をそのまま試す

### 目的

- Chrome上で動くか確認する
- 端末性能やカメラ癖を確認する

### 推奨技術構成

Step 1と同じ構成を使う。

#### 言語
- JavaScript

#### 画面
- 単純なHTML
- CSS
- Canvas

#### 実行環境
- Chrome on Android

#### 顔検出・顔ランドマーク
- MediaPipe Face Landmarker

#### 信号処理
- Step 1のJavaScript実装をそのまま利用

### 実装ステップ

#### 2-1. Web版をTabletで起動する
- Chromeでページを開く
- カメラ権限を許可する
- 前面カメラが使えることを確認する

#### 2-2. 基本動作を確認する
- 顔が取れるか
- ROIが追従するか
- BPMが表示されるか
- 波形描画が極端に重くないか

#### 2-3. 端末依存の確認を行う
- FPSが安定するか
- 自動露出でROI輝度が大きく揺れないか
- 顔追跡が途切れないか
- 前面カメラの画角や画質で問題がないか

#### 2-4. 負荷を軽くする
必要なら次を調整する。

- 顔ランドマーク推定を毎フレームではなく間引く
- 波形描画頻度を落とす
- ROIサイズを見直す
- 処理対象解像度を下げる
- BPM更新周期を長めにする

#### 2-5. デバッグ表示を追加する
あると便利な表示は次の通り。

- 実測FPS
- ROIサイズ
- 顔検出の成否
- 信号品質の簡易指標
- 現在の解像度

### Step 2の役割

Step 2は、ネイティブ化する前に次を見極める段階である。

- Web版の設計で十分か
- Tabletのブラウザ上では厳しいか
- 問題の原因がアルゴリズムか実機依存か

### Step 2の分岐判断

#### Webのままで十分な場合
- 顔追跡が安定している
- BPMが妥当
- 負荷が許容範囲
- UIも十分

この場合は、しばらくWeb版を育ててもよい。

#### ネイティブ化を検討すべき場合
- FPSが不安定
- 自動露出や色変動の影響が大きい
- 顔追跡はできるがBPMが不安定
- ブラウザ依存の制約が強い
- 実機向けに安定性を上げたい

---

## Step 3: 必要ならAndroidネイティブ化

### 目的

- 安定化
- 最適化
- 実機向け調整

### 基本方針

Androidネイティブアプリでは、なるべく単純な技術構成を採用する。
最初から複雑なクロスプラットフォーム構成やC++導入は行わない。

### 推奨技術構成

#### 言語
- Kotlin

#### 開発環境
- Android Studio

#### UI
- XMLレイアウト
- まずは単一Activityで十分

#### カメラ
- CameraX

#### 顔検出
- ML Kit Face Detection
- 必要なら後でMediaPipeに切り替え検討

#### 信号処理
- Kotlinで自前実装

### Android側で単純に保つポイント

- 単一Activityで開始する
- 画面数を増やさない
- CameraXを使う
- 顔検出はまずML Kitで試す
- 信号処理はKotlinで書く
- ComposeやNDKは最初から導入しない

### 推奨ファイル構成のイメージ

- MainActivity.kt
- CameraController.kt
- FaceProcessor.kt
- SignalProcessor.kt
- OverlayView.kt
- activity_main.xml

### 各ファイルの役割

#### MainActivity.kt
- 画面初期化
- CameraX起動
- 全体制御

#### CameraController.kt
- カメラバインド
- 前面カメラ選択
- ImageAnalysis設定

#### FaceProcessor.kt
- 顔検出
- ROI決定
- ROI安定化

#### SignalProcessor.kt
- RGB時系列バッファ
- 正規化
- detrend
- バンドパス
- BPM推定

#### OverlayView.kt
- 顔枠描画
- ROI描画
- デバッグ表示

### 実装ステップ

#### 3-1. CameraXで前面カメラを表示する
- Previewを表示する
- ImageAnalysisを有効にする
- フレームを処理パイプラインへ流す

#### 3-2. 顔検出を入れる
- ML Kit Face Detectionで顔領域を取る
- まずは顔矩形ベースでもよい
- 必要ならROIの取り方を精密化する

#### 3-3. ROI平均色を取得する
- ImageAnalysisで受け取ったフレームからROIを切り出す
- ROIの平均RGBを計算する
- 時系列バッファに保存する

#### 3-4. SignalProcessorを移植する
- Step 1で確立した処理手順をKotlinへ移植する
- 処理内容を変えすぎない
- まずはWeb版と同じアルゴリズムで揃える

#### 3-5. BPMと波形を表示する
- BPM数値を表示する
- 必要なら簡易波形ビューを描画する
- 顔枠とROIも表示する

#### 3-6. 安定化を行う
- 処理解像度を最適化する
- 推論頻度を見直す
- ROI位置の揺れを抑える
- BPM更新を安定化する

### Step 3で最初は入れないもの

- Flutter
- React Native
- NDK
- TensorFlow Lite
- Jetpack Compose
- 複数画面構成
- 複雑なDI構成

これらは必要になってから追加すればよい。

---

## 実装順序のまとめ

### フェーズ1
- HTMLとJavaScriptでPoCを作る
- PCのWebカメラで顔追跡とBPM推定を成立させる

### フェーズ2
- 同じWeb版をTabletのChromeで動かす
- 実機での性能と安定性を観察する

### フェーズ3
- 必要な場合だけAndroidネイティブへ移行する
- Kotlin + CameraX + ML Kit + 自前信号処理で最小構成を組む

---

## 最終的なおすすめ

最初から完成形を作ろうとせず、次の順番を守るのがよい。

1. まずは単純なHTMLでWeb PoCを成立させる
2. TabletのChromeでそのまま検証する
3. 問題があればAndroidネイティブ化する
4. ネイティブ化しても、Kotlin + CameraX + ML Kitの単純構成を維持する

この進め方なら、無駄な実装を増やさずに、

- アルゴリズムの妥当性確認
- 実機での成立性確認
- 必要最小限のネイティブ最適化

を順番に進められる。