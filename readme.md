# 概要

競馬予想のAIです。  
スクレイピングによるデータ収集～データを使ってのモデルの学習までを行っています。

## 環境

### pythonバージョン

3.11.7

### パッケージ  

（主要なものだけ記載しています。詳しくはrequirements.txtを見てください）  
|パッケージ名|バージョン|
|-----|-----|
|pandas|2.1.4|
|beautifulsoup4|4.12.3|
|category-encoders|2.6.3|
|numpy|1.26.3|
|scikit-learn|1.3.2|

>[!NOTE]
>requirements.txtを使ってパッケージをインストールする場合、pytorch関連については環境に合わせて個別でインストールしてください  
>[pytorch公式](https://pytorch.org/get-started/locally/)

## サンプルデータ

data/配下に保存
ファイル数が多いため、リポジトリには各種１個ずつgitの管理対象としている

.pkl：pickle形式で保存したデータフレーム
.pickle：pickle 形式で保存したデータフレーム以外のオブジェクト

source py311/Scripts/activate

## 各ディレクトリ・ファイルの説明

### [models](./models/)  

作成したモデルを格納

### [Processed-Data](./Processed-Data/)

加工済みのデータを格納  

### [Raw-Data](./Raw-Data/)

スクレイピングで入手したデータを格納  

### [src](./src/)

ソースコードを格納  

### [template](./template/)

データの加工に使用したデータフレームのひな形を格納  
