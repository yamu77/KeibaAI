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
- 距離はコースの種類と距離に分ける（距離については数値のスケールを小さくするため1/100にしています）
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
ワンホットエンコーディング用のモデルを作成するときは定性データだけを全量分抽出したものを用意して作成しています。

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
データではなくファイルパスだけ渡すのは、将来的にデータが増えるとメモリが足りなくなる可能性があるためです。

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

学習の前にオプティマイザを使用してパラメータを探索します。

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

### 2-1.テンプレートを作成

親馬の過去成績では、全成績を競馬場、距離、クラス等で分類したうえで集計します。

```python
class HorseProcessor:
    def remove_str(x: any) -> str:
        x_str = str(x)
        is_contain_num = re.search(r"\d+", x_str)
        if is_contain_num:
            return is_contain_num.group()
        else:
            return "0"

    @staticmethod
    def transform_race_length(length: str | int) -> str:
        """距離を変換する

        Args:
            length (str | int): 距離

        Raises:
            TypeError: 型が不正

        Returns:
            str: 変換後の距離
        """
        if isinstance(length, str):
            length = int(length)
        elif math.isnan(length):
            length = 0
        match length:
            case length if length < 1000:
                return "不明"
            case length if length <= 1300:
                return "S"
            case length if length <= 1899:
                return "M"
            case length if length <= 2100:
                return "I"
            case length if length <= 2700:
                return "L"
            case length if length > 2700:
                return "E"

    @staticmethod
    def transform_held(held: str | int) -> str:
        """開催地を変換する

        Args:
            held (str | int): 開催地

        Returns:
            str: 変換後の開催地
        """
        if isinstance(held, int):
            held = str(held)
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
    def transform_race_name(race: str | int) -> str:
        """レース名を変換する

        Args:
            race (str | int): レース名

        Returns:
            str: 変換後のレース名
        """
        # r"新馬|未勝利|1勝|2勝|3勝|オープン"
        if isinstance(race, int):
            race = str(race)
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
        df_processed = df[["開催", "天気", "レース名", "着順", "距離", "馬場"]]
        return df_processed

    @staticmethod
    def divide_corse(df: pd.DataFrame) -> pd.DataFrame:
        """コースの種類と距離を分ける

        Args:
            df (pd.DataFrame): 加工前データ

        Returns:
            pd.DataFrame: 加工後データ
        """
        df_divided = df
        df_divided["コース"] = df_divided["距離"].map(
            lambda x: x[0] if not isinstance(x, int) else 0
        )
        df_divided["距離"] = df_divided["距離"].map(
            lambda x: x[1:] if not isinstance(x, int) else 0
        )
        return df_divided

    @staticmethod
    def delete_invalid_race(df: pd.DataFrame) -> pd.DataFrame:
        """失格や中止になったレースを削除する

        Args:
            df (pd.DataFrame): 加工前データ

        Returns:
            pd.DataFrame: 加工後データ
        """
        df = df.drop(index=df[df["着順"] == 0].index)
        df = df.drop(index=df[df["着順"] == "0"].index)
        return df

    @classmethod
    def process(cls, path: pd.DataFrame | str):
        """データを整形する

        Args:
            path (pd.DataFrame | str): データのパス

        Returns:
            pd.DataFrame: 整形後データ
        """
        if isinstance(path, str):
            df_raw = pd.read_pickle(path)
        elif isinstance(path, pd.DataFrame):
            df_raw = path
        # 欠損値を0埋め
        df_processed = df_raw.fillna(0)
        # カラム名の空白を削除
        df_processed.columns = df_processed.columns.str.replace(" ", "")

        df_processed = cls.drop_columns(df_processed)
        df_processed["開催"] = df_processed["開催"].map(cls.transform_held)
        df_processed["レース名"] = df_processed["レース名"].map(cls.transform_race_name)
        df_processed = cls.divide_corse(df_processed)
        df_processed["距離"] = df_processed["距離"].map(cls.transform_race_length)
        df_processed["着順"] = df_processed["着順"].map(cls.remove_str)
        df_processed = cls.delete_invalid_race(df_processed)
        df_processed = df_processed.replace(0, "不明")
        return df_processed.iloc[::-1].reset_index(drop=True)

dir_list = os.listdir("../Raw-Data/Pedigree/")
df_integrated = pd.DataFrame()
for i in tqdm(dir_list):
    with open(f"../Raw-Data/Pedigree/{i}", "rb") as f:
        peds = pickle.load(f)
        for ped in peds:
            try:
                df = pd.read_pickle(f"../Raw-Data/Pedigree-Results/{ped}.pkl")
                df = HorseProcessor.process(df)
                df_integrated = pd.concat([df_integrated, df])
                df_integrated = df_integrated.reset_index(drop=True)
            except Exception as e:
                print(ped)
                raise Exception(e)
```

この処理で親馬のテンプレート作成に必要なデータを抽出します。

| 開催 | 天気 | レース名 | 着順 | 距離 | 馬場 | コース |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 東京 | 晴 | OP | 3 | E | 良 | 芝 |
| 東京 | 晴 | OP | 6 | L | 良 | 芝 |
| 中山 | 晴 | OP | 3 | L | 良 | 芝 |

抽出したデータはこのようになっています。  
あとは、このデータが問題なく集計できるようにデータフレームのテンプレートを作成していきます。  
今回は分類に開催、レースのクラス(重賞または非重賞)、距離(SMILE区分)、馬場、コースの5つです。  
これらで分類したうえで1着,2着,3着,4着以下の回数をそれぞれ集計します。出走回数はそれぞれの回数を足せば出せる情報になるのであえて集計していません。  
  
テンプレートは以下の方法で作成します。

```python
place = [
    "札幌",
    "函館",
    "福島",
    "新潟",
    "中山",
    "東京",
    "中京",
    "京都",
    "阪神",
    "小倉",
    "その他",
]
race = ["重賞", "非重賞"]
length = ["S", "M", "I", "L", "E"]
state = ["良", "稍", "重", "不"]
seed = ["芝", "ダ", "障"]
win = ["1", "2", "3", "3<"]
columns = []
for p in place:
    for r in race:
        for l in length:
            for s in seed:
                for se in state:
                    for wi in win:
                        columns.append(f"{p}_{r}_{l}_{s}_{se}_{wi}")

columns = [i for i in columns if not re.match(r"札幌_重賞_L.*", i)]
columns = [i for i in columns if not re.match(r"札幌_重賞_(S|I)_ダ.*", i)]
columns = [i for i in columns if not re.match(r"札幌_非重賞_L_ダ.*", i)]
columns = [i for i in columns if not re.match(r"札幌_非*重賞_(S|M|I|L)_障.*", i)]
columns = [i for i in columns if not re.match(r"札幌_非*重賞_E.*", i)]

columns = [i for i in columns if not re.match(r"函館_重賞_[SI]_ダ.*", i)]
columns = [i for i in columns if not re.match(r"函館_重賞_[LE].*", i)]
columns = [i for i in columns if not re.match(r"函館_非重賞_[IL]_ダ.*", i)]
columns = [i for i in columns if not re.match(r"函館_非重賞_E.*", i)]
columns = [i for i in columns if not re.match(r"函館_非*重賞_._障.*", i)]

columns = [i for i in columns if not re.match(r"福島_重賞_(S|L|E).*", i)]
columns = [i for i in columns if not re.match(r"福島_重賞_(M|I)_[ダ障].*", i)]
columns = [i for i in columns if not re.match(r"福島_重賞_._障.*", i)]
columns = [i for i in columns if not re.match(r"福島_非重賞_(I|L)_ダ.*", i)]
columns = [i for i in columns if not re.match(r"福島_非重賞_[SMIL]_障.*", i)]
columns = [i for i in columns if not re.match(r"福島_非重賞_E_[芝ダ].*", i)]

columns = [i for i in columns if not re.match(r"新潟_重賞_S_[ダ].*", i)]
columns = [i for i in columns if not re.match(r"新潟_重賞_[SMIL]_障.*", i)]
columns = [i for i in columns if not re.match(r"新潟_非*重賞_[IL]_ダ.*", i)]
columns = [i for i in columns if not re.match(r"新潟_非*重賞_E_[芝ダ].*", i)]
columns = [i for i in columns if not re.match(r"新潟_非重賞_[SMIL]_障.*", i)]

columns = [i for i in columns if not re.match(r"中山_重賞_[ILE]_ダ.*", i)]
columns = [i for i in columns if not re.match(r"中山_重賞_[SMIL]_障.*", i)]
columns = [i for i in columns if not re.match(r"中山_非重賞_[IE]_ダ.*", i)]
columns = [i for i in columns if not re.match(r"中山_非重賞_[E]_芝.*", i)]
columns = [i for i in columns if not re.match(r"中山_非重賞_[SMI]_障.*", i)]

columns = [i for i in columns if not re.match(r"東京_非*重賞_[SMIL]_障.*", i)]
columns = [i for i in columns if not re.match(r"東京_非*重賞_S_芝.*", i)]
columns = [i for i in columns if not re.match(r"東京_重賞_[LE]_ダ.*", i)]
columns = [i for i in columns if not re.match(r"東京_非重賞_[E]_ダ.*", i)]

columns = [i for i in columns if not re.match(r"中京_重賞_._障.*", i)]
columns = [i for i in columns if not re.match(r"中京_重賞_[SIE]_ダ.*", i)]
columns = [i for i in columns if not re.match(r"中京_重賞_[L]_芝.*", i)]
columns = [i for i in columns if not re.match(r"中京_非重賞_[IE]_ダ.*", i)]
columns = [i for i in columns if not re.match(r"中京_非重賞_E_芝.*", i)]
columns = [i for i in columns if not re.match(r"中京_非重賞_[SMIL]_障.*", i)]

columns = [i for i in columns if not re.match(r"京都_非*重賞_[LE]_ダ.*", i)]
columns = [i for i in columns if not re.match(r"京都_非*重賞_[SMIL]_障.*", i)]

columns = [i for i in columns if not re.match(r"阪神_重賞_[SLE]_ダ.*", i)]
columns = [i for i in columns if not re.match(r"阪神_重賞_[SMIL]_障.*", i)]
columns = [i for i in columns if not re.match(r"阪神_非重賞_[SMIL]_障.*", i)]
columns = [i for i in columns if not re.match(r"阪神_非重賞_L_ダ.*", i)]
columns = [i for i in columns if not re.match(r"阪神_非重賞_E_[ダ芝].*", i)]

columns = [i for i in columns if not re.match(r"小倉_重賞_[SMI]_[ダ障].*", i)]
columns = [i for i in columns if not re.match(r"小倉_重賞_L_[芝障].*", i)]
columns = [i for i in columns if not re.match(r"小倉_重賞_E_[芝ダ].*", i)]
columns = [i for i in columns if not re.match(r"小倉_非重賞_[SMIL]_障.*", i)]
columns = [i for i in columns if not re.match(r"小倉_非重賞_[IL]_ダ.*", i)]
columns = [i for i in columns if not re.match(r"小倉_非重賞_E_[ダ芝].*", i)]

columns = [i for i in columns if not re.match(r"その他_非*重賞_._障.*", i)]
df_tmp = pd.DataFrame(columns=columns, index=[0]).fillna(0)
df_tmp
```

力技ですが、全通り分の組み合わせを作成して不要な項目を除外しています。  
不要な項目化は実際に集計して0だったものを削除していきます。  
上記のコードは除外できる項目をすべて除外するように調整されたものになります。

```python
df = pd.read_pickle("../tmp/peds-results.pkl")
race = ["G3", "G1", "G2"]
for index, row in df.iterrows():
    col = [row["距離"], row["コース"]]
    if "不明" in col:
        continue
    w = row["着順"] if int(row["着順"]) <= 3 else "3<"
    s = row["馬場"] if row["馬場"] != "不明" else "良"
    r = "重賞" if row["レース名"] in race else "非重賞"
    col = f'{row["開催"]}_{r}_{row["距離"]}_{row["コース"]}_{s}_{w}'
    df_tmp[col] += 1

for i in df_tmp:
    tex = df_tmp[i][0]
    if tex == 0:
        print(i)
```

これで0のものが洗い出せます。（最初は表示数が多くなってしまうので正規表現などで表示する項目を絞って少しずつ除外していきました。）

### 2-2. PCIによるデータの圧縮

不要な項目は除外しましたが、それでも項目数が多いため、PCIを用いてデータの圧縮を行います。  

#### 2-2-1. クラス

```python
class PedigreePCA:
    def __init__(self, n_components: float, path: str):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.model_path = path

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """訓練データに対してPCAを実行し、累積寄与率と各成分の寄与率を計算する

        Args:
            df (pd.DataFrame): 訓練データ

        Returns:
            pd.DataFrame: 変換後のデータ
        """
        transformed_data = self.pca.fit_transform(df)
        print(f"累積寄与率: {self.pca.explained_variance_ratio_.sum()}")
        for i, ratio in enumerate(self.pca.explained_variance_ratio_, start=1):
            print(f"成分{i}の寄与率: {ratio}")
        return pd.DataFrame(transformed_data)

    def save_model(self):
        """モデルの保存"""
        with open(self.model_path, "wb") as f:
            pickle.dump(self.pca, f)
        print(f"モデルを{self.model_path}に保存しました。")

    def load_model(self):
        """モデルの読み込み"""
        with open(self.model_path, "rb") as f:
            self.pca = pickle.load(f)
        print(f"モデルを{self.model_path}から読み込みました。")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """保存されたモデルを使用して、テストデータに対してPCAを実行する

        Args:
            df (pd.DataFrame): テストデータ

        Returns:
            pd.DataFrame: 変換後のデータ
        """
        transformed_data = self.pca.transform(df)
        return pd.DataFrame(transformed_data)
```

#### 2-2-2. 説明

コンストラクタでは圧縮後のデータの項目数とPCIに用いるデータのファイルパスを指定しています。  
fit_transform関数でPCAの訓練と成分数ごとの累積寄与率と各成分の寄与率を表示しています。  
寄与率はその成分が元データの情報をどれくらい持っているかの割合で、全ての成分の寄与率を足し合わせたものが累積寄与率です。  
累積寄与率が1に近いほど元データに近い情報を持っていることになります。  
今回は60%以上の累積寄与率を条件にしました。  

条件を満たすデータの項目数が見つかったらsave_model関数でモデルを保存します。

## 3.レース情報&出走馬の情報

レース情報、出走馬の情報では

- 着順
- 枠番
- 馬番
- 斤量
- 性別
- 年齢
- 体重
- 増減
- 3着以内

のデータをレース結果から抽出して加工します。

### 3-1. クラス

```python
class RaceResults:
    """レース結果をデータを整形する"""

    with open("../models/race_info_encoder.pickle", "rb") as f:
        encoder = pickle.load(f)

    with open("../models/horse_info_scaler.pickle", "rb") as f:
        scaler = pickle.load(f)

    def read_df(path: str) -> pd.DataFrame:
        """データフレームの読み込み

        Args:
            path (str): pickleのパス

        Raises:
            TypeError: 引数が文字列でなければエラーを出す

        Returns:
            pd.DataFrame: 読み込んだデータフレーム
        """
        if not isinstance(path, str):
            raise TypeError(
                f'"path" argument is expected to be of type str, got {type(path).__name__} instead'
            )
        results_processed = pd.read_pickle(path)
        return results_processed

    def divide_weight_gender(df_raw: pd.DataFrame) -> pd.DataFrame:
        """性齢の値を性別と年齢に分け、馬体重も体重と増減に分ける。性別はダミー変数化する

        Args:
            df_raw (pd.DataFrame): 対象データフレーム

        Returns:
            pd.DataFrame: 変換後データフレーム
        """
        df = df_raw.copy()
        gender = df["性齢"].str[0]
        df["牡"] = gender.map(lambda x: 1 if x == "牡" else 0)
        df["牝"] = gender.map(lambda x: 1 if x == "牝" else 0)
        df["セ"] = gender.map(lambda x: 1 if x == "セ" else 0)
        df["年齢"] = df["性齢"].str[1:]
        df["体重"] = df["馬体重"].replace(
            to_replace=r"(\d+).*", value=r"\1", regex=True
        )
        df["増減"] = df["馬体重"].replace(
            to_replace=r"\d+\(\+{0,1}([-]{0,1}\d+)\)", value=r"\1", regex=True
        )
        return df

    def transform_rank(df_raw: pd.DataFrame, multi=True) -> pd.DataFrame:
        """着順のデータを３着以内かどうかの値にする。(3着以内であれば1、そうでなければ0)

        Args:
            df_raw (pd.DataFrame): 対象データフレーム

        Returns:
            pd.DataFrame: 変換後データ
        """
        df = df_raw.copy()
        threshold = 3**multi
        df["3着以内"] = df["着順"].apply(
            lambda x: 1 if isinstance(x, int) and x <= threshold else 0
        )
        return df

    def drop_columns(df_raw: pd.DataFrame, columns: [str]) -> pd.DataFrame:
        """不要なカラムを削除する

        Args:
            df_raw (pd.DataFrame): 対象データフレーム
            columns (str]): 削除するカラム名

        Returns:
            pd.DataFrame: 削除後データフレーム
        """
        df = df_raw.drop(columns=columns)
        return df

    def transform_date(date: str) -> str:
        """日付を変換して、その年の1月1日からの週数を計算する

        Args:
            date (str): 日付の文字列（%Y年%m月%d日）

        Returns:
            str: 変換後の日付文字列
        """
        # 日付の形式を変換
        date_converted = datetime.datetime.strptime(date, "%Y年%m月%d日")
        # その年の1月1日を計算
        base_date = datetime.datetime(date_converted.year, 1, 1)
        # 週数の差を計算
        return (date_converted - base_date).days // 7

    def extraction_drop_columns(
        df: pd.DataFrame, columns: [str]
    ) -> (pd.DataFrame, pd.DataFrame):
        """データフレームをカラム指定で分割する

        Args:
            df (pd.DataFrame): 対象のデータフレーム
            pd ([str]): 分割するカラム名

        Returns:
            pd.DataFrame, pd.DataFrame): 指定したカラムを抽出したデータフレームと、それを取り除いたデータフレーム
        """
        df_extraction = df.loc[:, columns]
        df_dropped = df.drop(columns=columns)
        return df_extraction, df_dropped

    def add_rows(df_raw: pd.DataFrame, rows: int) -> pd.DataFrame:
        df = df_raw.copy()
        df = pd.concat(
            [
                df,
                pd.DataFrame(np.zeros((rows, len(df.columns))), columns=df.columns),
            ],
            ignore_index=True,
        )
        return df

    @classmethod
    def adapt_race_info(cls, df_raw: pd.DataFrame) -> pd.DataFrame:
        """レース情報の日付をその年の週数に、コースの長さのスケールを1/100にする。データ型も変更する

        Args:
            df_raw (pd.DataFrame): レース情報

        Returns:
            pd.DataFrame: 変換後のデータ
        """
        df = df_raw.loc[[0], :]
        df["date"] = cls.transform_date(df.loc[0, "date"])
        df["course_length"] = float(df.loc[0, "course_length"]) / 100
        df["round"] = df["round"].astype(float)

        df = cls.encoder.transform(df)
        return df

    @classmethod
    def horse_info_transform(cls, df_raw: pd.DataFrame) -> pd.DataFrame:
        """出走馬情報の標準化と足りない行の補填、型変換をする

        Args:
            df_raw (pd.DataFrame): 出走馬情報

        Returns:
            pd.DataFrame: 変換後のデータ
        """
        df = df_raw.copy()
        columns_to_scale = ["体重", "増減"]
        df[columns_to_scale] = cls.scaler.transform(df[columns_to_scale])
        shortage_rows = 18 - len(df)
        df = cls.add_rows(df, shortage_rows)
        df["年齢"] = df["年齢"].astype(float)
        return df

    @classmethod
    def make_infos(cls, path: str, multi=True) -> {pd.DataFrame or str}:
        """レース結果をレース情報、出走馬情報、出走馬ID、レース日付、正解ラベルの5個に分ける

        Args:
            path (str): レース結果ファイルのパス

        Returns:
            {pd.DataFrame or str}: dictで保存。キーはそれぞれrace,horse,ids,date,label。date以外はデータフレーム
        """
        drop_columns = [
            "馬名",
            "性齢",
            "騎手",
            "タイム",
            "着差",
            "人気",
            "調教師",
            "単勝",
            "jockey_id",
            "馬体重",
            "着順",
        ]
        race_info_columns = [
            "date",
            "round",
            "course_length",
            "course_type",
            "course_way",
            "weather",
            "state_grass",
            "state_dirt",
            "place",
            "class",
        ]
        df_raw = cls.read_df(path)
        df = df_raw.copy()
        # データの0埋めを行う
        df = df.fillna(0)
        # 馬体重のカラムについては「0(0)」で埋める
        df["馬体重"].replace(0, "0(0)", inplace=True)
        df["馬体重"].replace("計不", "0(0)", inplace=True)
        df = cls.divide_weight_gender(df)
        df = cls.transform_rank(df, multi=multi)
        df = cls.drop_columns(df, drop_columns)
        race_info, horse_info = cls.extraction_drop_columns(df, race_info_columns)
        horse_id, horse_info = cls.extraction_drop_columns(horse_info, ["horse_id"])

        # 標準化等の変換
        race_info = cls.adapt_race_info(race_info)
        horse_info = cls.horse_info_transform(horse_info)
        return {
            "race": race_info,
            "horse": horse_info.drop(["3着以内"], axis=1),
            "ids": list(horse_id.iloc[:, 0].values),
            "date": df_raw.loc[0, "date"],
            "label": horse_info["3着以内"],
        }
```

### 3-2. 説明

抽出したデータの内、着順の値については今回の予想では3着以内かどうかの推論を行うので3着以内であれば1、そうでなければ0に変換しています。  
一応make_infos関数の引数multiをFalseにすれば1着のみ1,それ以外を0に変換できます。  
それ以外の日付や開催地、馬体重などの加工については出走馬の過去成績と同じようなことをしているので省略します。  
最後の値を返すときは

- レースの情報
- 出走馬の情報
- 出走馬のID
- レース日付
- 正解ラベル

に分けた辞書型にしています。

## 4.最終的な学習データの作成

1~3の処理を使用して学習データを作成します。
最終的には

1. 正解ラベル(3着以内かどうか)
2. レース情報
3. 出走馬情報
4. 成績（出走馬とその親馬の成績を連結）

の4種類のデータを辞書型にして保存します。

### 4-1. 処理

```python
def df_to_tensor_1d(df_raw: pd.DataFrame) -> torch.Tensor:
    """データフレームを1次元のテンソルに変換する
    Args:
        df_raw (pd.DataFrame): 変換前データフレーム
    Returns:
        torch.Tensor: 変換後テンソル
    """
    df = df_raw.copy()
    df_array = df.values.flatten()
    return torch.tensor(df_array, dtype=torch.float32)


def add_tensor(tensors: list[torch.Tensor]) -> torch.Tensor:
    """テンソルを結合する
    Args:
        tensors (list[torch.Tensor]): 結合するテンソル
    Returns:
        torch.Tensor: 結合後テンソル
    """
    tensors_tmp = tensors
    add_num = 18 - len(tensors)
    for _ in range(add_num):
        array_zeros = torch.zeros(40)
        tensors_tmp.append(array_zeros)
    return torch.cat(tensors_tmp)


def make_train_data(path: str, multi=True):
    """データを作成する
    Args:
        path (str): データのパス
        multi (bool, optional): 単勝かどうか。 Defaults to True.
    Returns:
        dict: データ
    """
    vae = VAE("../models/horse_result_VAE.pth")
    data = RaceResults.make_infos(path, multi=multi)
    ped_paths = [f"../Raw-Data/Pedigree/{i}.pickle" for i in data["ids"]]
    results_paths = [f"../Raw-Data/Horse-Results/{i}.pkl" for i in data["ids"]]
    ped_raw = PedigreeResults.process(ped_paths)
    result_raw = HorseResult.process(results_paths)
    vae_raw = vae.process(result_raw)
    vae_result = add_tensor(vae_raw)
    race = df_to_tensor_1d(data["race"])
    horse = df_to_tensor_1d(data["horse"])
    peds = df_to_tensor_1d(ped_raw)
    label = df_to_tensor_1d(data["label"])
    return {
        "label": label,  # 正解ラベル
        "race": race,  # レース情報
        "horse": horse,  # 出走馬
        "results": torch.cat([vae_result, peds]),  # 成績
    }
```

### 4-2. 説明

上半分の処理は生データの加工です。  
下半分では

- データの行数が18未満の場合は18行になるように全て0の行を追加
- データフレーム型を１次元のテンソル型に変換
- 辞書型で4個のデータにして保存

を行っています。
