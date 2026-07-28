# Web Heart Rate Monitor
本リポジトリでは、カメラ映像の顔から心拍を計測するWebアプリを開発しています。

https://sakamo-bnn.github.io/face-rppg-web/

## 概要
一般的な心拍の測定はPPGセンサを用いており、測定機器との接触を要します。従って、不特定多数の人が使用する環境では、他者が装着した機器を身につけることへの不快感や、衛生面への懸念が生じます。

本アプリでは、カメラ映像から、拍動に伴う顔色の微細な変動を用いて心拍を計測します。これにより、専用のセンサを装着することなく、非接触かつ手軽に心拍をリアルタイムに測定できます。 

## 仕組み
<iframe
  src="./docs/face-rppg-web（A4縦）.pdf"
  width="100%"
  height="700px">
</iframe>


[![PDFプレビュー](./docs/face-rppg-web（A4縦）.pdf)](./docs/face-rppg-web（A4縦）.pdf)

## ローカルにおける起動方法
ローカルで`index.html`をそのまま開くのではなく、サーバを立てる方が安定して動作します。

VSCodeのLive Serverの利用を推奨しますが、Pythonを使って簡易サーバを立てることも可能です。

```bash
cd web_heartrate_app
python -m http.server 8000
```

その後ブラウザで `http://localhost:8000` を開いてください。