３代目以降の競馬予想AIを作っていく

# 環境
python 3.12.0
(あとでrequirements.txtを作ってパッケージの情報をまとめる)

# サンプルデータ
data/配下に保存
ファイル数が多いため、リポジトリには各種１個ずつgitの管理対象としている

.pkl：pickle形式で保存したデータフレーム
.pickle：pickle 形式で保存したデータフレーム以外のオブジェクト

source py311/Scripts/activate

# ファイルの説明
## ./models/
### horse_result_encoder.pickle  
過去成績に対してダミー変数化をするためのモデル  
### horse_result_VAE.pth  
過去成績に対してVAEによる変換をするためのモデル  
### horse_results_scaler.pickle  
過去成績に対して標準化をするためのモデル  
### pedigree_pca.pickle  
親データに対してPCAをするためのモデル  
### race-info-encoder.pkl  
レース情報に対してダミー変数化をするためのモデル  
