# 概要

AIを作るにあたって作成したモデルを格納

## 各モデル説明

### horse_info_scaler.pickle

レース出走馬の情報を標準化するためのモデル  

### horse_result_encoder.pickle

過去成績に対してダミー変数化を行うone-hot-encodingモデル  

### horse_result_VAE.pth

過去成績に対してデータの圧縮を行うためのVAEモデル  

### horse_results_scaler.pickle

過去成績に対して標準化をするためのモデル  

### pedigree_pca.pickle

親データに対して次元圧縮を行うためのPCAモデル  

### race-info-encoder.pkl

レース情報のデータに対してダミー変数化を行うone-hot-encodingモデル  
