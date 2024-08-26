# 元データ→学習データの加工

過去レース結果をもとに学習データを作成していきます。

- 出走馬の過去成績
- 親馬の過去成績
- レース情報

の3つに対して別々に加工を行い、それらを最終的に統合して一つの変換処理としました。

## 1.出走馬の過去成績

出走馬の過去成績では不要なデータの削除や軽い加工を行った後VAEによる変換を行います

### 1-1.VAEの学習用データ作成クラス

```python
class HorseProcessor:
    max_rows = 10
    columns_to_scale = ["馬体重", "増減", "斤量"]
    with open("../models/horse_result_encoder.pickle", "rb") as f:
        encoder: ce.OneHotEncoder = pickle.load(f)
    with open("../models/horse_results_scaler.pickle", "rb") as f:
        scaler: StandardScaler = pickle.load(f)

    def remove_str(x: any) -> str:
        """文字列を抽出する
        Args:
            x (any): 文字列
        Returns:
            str: 抽出した文字列
        """
        x_str = str(x)
        is_contain_num = re.search(r"\d+", x_str)
        if is_contain_num:
            return is_contain_num.group()
        else:
            return "0"

    def convert_date(x: str | int) -> int:
        """日付をその年の1日1月を基点とした週数に変換する

        Args:
            x (str | int): 日付(YYYY/MM/DD)

        Returns:
            int: 日数
        """
        # 日付の形式を変換
        date_converted = datetime.datetime.strptime(x, "%Y/%m/%d")
        # その年の1月1日を計算
        base_date = datetime.datetime(date_converted.year, 1, 1)
        # 週数の差を計算
        return (date_converted - base_date).days // 7

    @staticmethod
    def transform_held(held: str) -> str:
        """開催地を変換する
        Args:
            held (str): 開催地
        Returns:
            str: 変換後の開催地
        """
        trim_held = re.sub(r"\d*", "", held)
        if not trim_held in [
            "東京",
            "中山",
            "中京",
            "阪神",
            "札幌",
            "函館",
            "福島",
            "新潟",
            "京都",
            "小倉",
        ]:
            return "その他"
        return trim_held

    @staticmethod
    def transform_race_name(race: str) -> str:
        """レース名を変換する
        Args:
            race (str): レース名
        Returns:
            str: 変換後のレース名
        """
        # r"新馬|未勝利|1勝|2勝|3勝|オープン"
        if re.search(r".*(新馬|未勝利|1勝|2勝|3勝|OP|G1|G2|G3|L).*", race):
            transform_name = re.sub(
                r".*(新馬|未勝利|1勝|2勝|3勝|OP|G1|G2|G3|L).*", r"\1", race
            )
        else:
            transform_name = "その他"
        return transform_name

    @staticmethod
    def extract_addition(df: pd.DataFrame) -> pd.DataFrame:
        """体重の増減を抽出する

        Args:
            df (pd.DataFrame): 変化対象のデータ

        Returns:
            pd.DataFrame: 変換後のデータ
        """
        weight = df["馬体重"]
        addition = weight.map(lambda x: re.sub(r".*\(([+-]\d{1,3}|0)\).*", r"\1", x))
        addition = addition.map(lambda x: re.sub(r"\+", "", x))
        return addition

    @staticmethod
    def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
        """不要なカラムを削除

        Args:
            df (pd.DataFrame): 成績データ

        Returns:
            pd.DataFrame: 削除後データ
        """
        df_processed = df.drop(
            [
                "賞金",
                "厩舎ｺﾒﾝﾄ",
                "備考",
                "勝ち馬(2着馬)",
                "着差",
                "ﾀｲﾑ指数",
                "通過",
                "ペース",
                "上り",
                "馬場指数",
                "タイム",
                "映像",
                "騎手",
                "オッズ",
                "人気",
            ],
            axis=1,
        )
        return df_processed

    @staticmethod
    def divide_corse(df: pd.DataFrame) -> pd.DataFrame:
        """距離データをコースの種類と距離に分ける
        Args:
            df (pd.DataFrame): 加工前データ
        Returns:
            pd.DataFrame: 加工後データ
        """
        df_divided = df
        df_divided["コース"] = df_divided["距離"].map(lambda x: x[0])
        df_divided["距離"] = df_divided["距離"].map(lambda x: int(x[1:]) / 100)
        return df_divided

    @classmethod
    def divide_horse_weight(cls, df: pd.DataFrame) -> pd.DataFrame:
        """馬体重を分ける

        Args:
            df (pd.DataFrame): 加工前データ

        Returns:
            pd.DataFrame: 加工後データ
        """
        df_divided = df
        df_divided["馬体重"] = df_divided["馬体重"].map(
            lambda x: x.replace("計不", "0(0)")
        )
        weight_addition = cls.extract_addition(df_divided)
        df_divided["増減"] = weight_addition
        df_divided["馬体重"] = df_divided["馬体重"].map(
            lambda x: re.sub(r"\([+-]*\d+\)", "", x)
        )
        return df_divided

    @classmethod
    def process(cls, path):
        """データを整形する
        Args:
            path (str): データのパス
        Returns:
            pd.DataFrame: 整形後データ
        """
        df_raw = pd.read_pickle(path).head(cls.max_rows)
        df_processed = df_raw.copy()
        # カラム名の空白を削除
        df_processed.columns = df_processed.columns.str.replace(" ", "")
        df_processed = cls.drop_columns(df_processed)
        df_processed = cls.divide_horse_weight(df_processed)
        df_processed["日付"] = df_processed["日付"].map(cls.convert_date)
        df_processed["開催"] = df_processed["開催"].map(cls.transform_held)
        df_processed["レース名"] = df_processed["レース名"].map(cls.transform_race_name)
        df_processed = cls.divide_corse(df_processed)
        # df_processed["距離"] = df_processed["距離"].map(lambda x: int(x) / 100)
        df_processed["馬番"] = df_processed["馬番"].map(lambda x: 0 if x > 18 else x)
        df_processed["着順"] = df_processed["着順"].map(cls.remove_str)
        # 欠損値の処理
        df_processed["馬場"] = (
            df_processed["馬場"].fillna("不明").infer_objects(copy=False)
        )
        df_processed["天気"] = (
            df_processed["天気"].fillna("不明").infer_objects(copy=False)
        )
        df_processed = df_processed.fillna(0).infer_objects(copy=False)
        # 型をintにする
        df = df_processed.astype({"R": int, "枠番": int})
        # 標準化
        df[cls.columns_to_scale] = cls.scaler.transform(df[cls.columns_to_scale])
        # ダミー変数化
        df = cls.encoder.transform(df)
        # 行数を調整
        if len(df) < cls.max_rows:
            rows_to_add = cls.max_rows - len(df)
            # すべての項目が0の行を作成
            additional_rows = pd.DataFrame(
                np.zeros((rows_to_add, len(df.columns))), columns=df.columns
            )
            # 追加の行をDataFrameに結合
            df = pd.concat([df, additional_rows], ignore_index=True)
        return df.iloc[::-1].reset_index(drop=True)
```

### 1-2.説明

このクラスでは

1. 不要なデータの削除
2. 日付や開催地などの項目ごとの簡単な加工
3. 欠損値の処理
4. 標準化
5. ワンホットエンコーディング
6. 行数の調整

を行っています。

2の日付や開催地などの項目ごとの簡単な加工では  

- 日付をその年の1日1月を基点とした週数に変換
- 国内の中央競馬の開催地以外の値を「その他」に変換
- レース名は新馬、未勝利、1勝、2勝、3勝、オープン、G1、G2、G3、Lといったクラスに変換(それ以外の値があった場合は「その他」に変換)
- 距離はコースの種類と距離に分ける
- 馬体重を体重と体重の増減で別項目に分ける

といったことをしています。

3の欠損値の処理では、馬場と天気で空欄になっているデータを「不明」にしています。

4の標準化では、体重とその増減、斤量を標準化しています。  
標準化とはデータを平均0、分散1になるような値に変換することを指します。  
詳しい説明は省きますが標準化をすることによってデータのスケールが揃い、モデルの学習効率や精度が向上するといわれています。

5のワンホットエンコーディングでは、開催地やレース名などの項目をワンホットエンコーディングしています。  
ワンホットエンコーディングとは、定性的なデータを定量的なデータにすることを指します。  
例えば、
|開催地|
|---|
|東京|
|中山|
|中京|

といったようなデータがあった場合、
|開催地_東京|開催地_中山|開催地_中京|
|---|---|---|
|1|0|0|
|0|1|0|
|0|0|1|

といったように変換を行い、入力データを数値にします。

6の行数の調整では競走馬の最新10レース分のデータを取っています。（10レースより少ない場合は全項目が0のデータで埋める）  
基本的に機械学習の入力データは固定長のデータである必要がありますが、レースの出走数は馬によってことなるのでこういった処置をしています。

### 1-3.VAEモデルの作成に使用するクラス

位置エンコーディングのクラス

```python
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
```

エンコーダ

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, nheads, nlayers, dropout=0.1):
        super(Encoder, self).__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            hidden_dim, nheads, hidden_dim, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output
```

デコーダ

```python
class Decoder(nn.Module):
    def __init__(self, hidden_dim, input_dim, nheads, nlayers, dropout=0.1):
        super(Decoder, self).__init__()
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        decoder_layers = nn.TransformerDecoderLayer(
            hidden_dim, nheads, hidden_dim, dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_decoder(src, src)
        output = self.decoder(output)
        return output
```

VAE

```python
class TransformerVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, nheads, nlayers, dropout=0.1):
        super(TransformerVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, nheads, nlayers, dropout)
        self.decoder = Decoder(hidden_dim, input_dim, nheads, nlayers, dropout)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.fc_out = nn.Linear(latent_dim, hidden_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, src):
        encoded = self.encoder(src)
        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        z = self.reparameterize(mu, log_var)
        z = self.fc_out(z)
        decoded = self.decoder(z)
        return decoded, mu, log_var

    def get_latent_val(self, src):
        encoded = self.encoder(src)
        val = self.fc_mu(encoded)
        return val
```

損失関数

```python
def vae_loss(recon_x, x, mu, log_var):
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_div
```

### 1-4.説明

モデルの作成は上述の通りVAEを使用しています。  
VAEとは、オートエンコーダー(AE)の一種で、

1. 入力データをエンコードし潜在変数を生成
2. 潜在変数をデコードし入力データを再構成
3. 入力データと再構成したデータの誤差が少なくなるようにモデルを学習

といったことを行うモデルです。
通常のAEとの違いは、潜在変数を確率分布として扱うところで柔軟性や解釈性がAEと比べて高いです。  
  
AEのエンコーダー・デコーダーに使用するモデルは自由に決められますが、今回は時系列データを扱うため

- RNN
- Transformer

で迷いましたが、より学習が高速なTransformerを採用しました。

### 1-5.モデルの学習

```python
class TimeSeriesDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # pickleファイルからデータフレームを読み込む
        df = pd.read_pickle(self.file_paths[idx])
        df = df.astype("float32")
        # データフレームをテンソルに変換
        data_tensor = torch.tensor(df.values, dtype=torch.float32)
        return data_tensor
```

```python
file_names = os.listdir("../Processed-Data/Horse-Results/")
file_paths = list(map(lambda x: "../Processed-Data/Horse-Results/" + x, file_names))
train_paths, test_paths = train_test_split(file_paths, test_size=0.3)
train_paths, val_paths = train_test_split(train_paths, test_size=0.2)

# カスタムデータセットのインスタンス化
train_dataset = TimeSeriesDataset(train_paths)
val_dataset = TimeSeriesDataset(val_paths)
test_dataset = TimeSeriesDataset(test_paths)

# データローダーの設定
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

学習用データはpickleファイルで保存しているので、ファイルパスだけ渡してデータローダー側でデータの読み込みとテンソル化を行っています。  

```python
def objective(trial):
    # ハイパーパラメータの設定
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [8 * i for i in range(2, 13)])
    latent_dim = trial.suggest_int("latent_dim", 2, 20, log=True)

    # モデルとオプティマイザの設定
    model = TransformerVAE(
        input_dim=67,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        nheads=8,
        nlayers=8,
        dropout=0.1,
    )
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # 訓練ループ
    for epoch in range(15):  # エポック数は適宜調整
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(batch)
            loss = vae_loss(recon_batch, batch, mu, log_var)
            loss.backward()
            optimizer.step()

    # 検証データセットでの性能評価
    # 簡単化のために最後の訓練損失を使用
    return loss.item()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)  # 試行回数は適宜調整

# 最適なハイパーパラメータを取得
best_params = study.best_params
best_value = study.best_trial.value
```

学習の前にオプティマイザを使用してパラメータを探索しています。

```python
# モデルのインスタンス化
model = TransformerVAE(
    input_dim=67,
    hidden_dim=64,
    latent_dim=4,
    nheads=8,
    nlayers=8,
    dropout=0.1,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003321518308021874)

# エポック数
num_epochs = 1000
# 評価を行うエポック数
eval_interval = 100

# 訓練
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(batch)
        loss = vae_loss(recon_batch, batch, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    # 一定のエポック数ごとに検証データセットでモデルを評価
    if epoch % eval_interval == 0 or epoch == num_epochs - 1:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch.to(device)
                recon_batch, mu, log_var = model(val_batch)
                loss = vae_loss(recon_batch, val_batch, mu, log_var)
                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)
        print(
            f"Epoch {epoch}, Train Loss: {train_loss / len(train_loader.dataset)}, Val Loss: {val_loss}"
        )
```

オプティマイザで得られた最適なパラメータを使用して学習を行っています。  
学習後はテストデータで性能を見て問題なければ`torch.save(model.state_dict(), "../models/horse_result_VAE.pth")`でモデルを保存します。

## 2.親馬の過去成績

## 3.レース情報

## 4.最終的な学習データの作成
