# 概要

競馬予想のAI  
スクレイピングによるデータ収集～データを使ってのモデルの学習まで

## 環境

### pythonバージョン

3.11.7

### パッケージ  

（主要なものだけ記載。詳しくはrequirements.txtを参照）  
|パッケージ名|バージョン|
|-----|-----|
|pandas|2.1.4|
|beautifulsoup4|4.12.3|
|category-encoders|2.6.3|
|numpy|1.26.3|
|scikit-learn|1.3.2|

>[!NOTE]
>requirements.txtを使ってパッケージをインストールする場合、pytorch関連については環境に合わせて個別でインストールする  
>[pytorch公式](https://pytorch.org/get-started/locally/)

## サンプルデータ

data/配下に保存  
ファイル数が多いため、リポジトリには各種１個ずつgitの管理対象としている  

pickleファイルの拡張子の使い分け  
.pkl：pickle形式で保存したデータフレーム  
.pickle：pickle 形式で保存したデータフレーム以外のオブジェクト  

## 各ディレクトリ・ファイルについて

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

## 実装内容

長いので複数のファイルに分けて記載しています。  
記載しているソースは簡略化のため抜粋していますので詳しい処理はソースコードを直接見てください。

### [1.学習データに使用する元データの用意](./scraping.md)

### [2.学習データの作成](./data_preparation.md)

### [3.モデルの作成](./model_creation.md)
