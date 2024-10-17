# 学習

## 1.データセットの作成

最初にデータセットクラスを作成します。

```python
class CustomDataSet(Dataset):

    def __init__(self, data, is_file=False):
        """
        Args:
            file_paths (list of str): 学習用データファイルのパスのリスト
        """
        self.data = data
        self.file = is_file

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.file:
            file_path = self.data[idx]
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            inputs = torch.cat([data["race"], data["horse"], data["results"]])
            labels = data["label"]
            return inputs, labels
        else:
            data_set = self.data[idx]
            return data_set["data"], data_set["label"]
```

getitem関数でデータの取り出し方を定義しており、labelに格納されているデータを出力データとして、それ以外を入力データとして取り出しています。  
データを丸っと渡した場合とデータを保存しているファイルパスを渡した場合の２通りを書いていますが返す値は同じです。

## 2.モデル定義

次に推論に使用するモデルのクラスを定義します。

```python
class NNClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        layers=[128, 128],
        dropout_rate=0.2,
        activation="mish",
        output_activation="sig",  # 変数名を変更して明確にします
    ):
        # モデル保存のために保存しておくパラメータ
        self.input_size = input_size
        self.output_size = output_size
        self.layer_size = layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.output_activation = output_activation

        super(NNClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleDict(
            {
                "relu": nn.ReLU(),
                "leaky_relu": nn.LeakyReLU(),
                "mish": nn.Mish(),
            }
        )
        self.dropout = nn.Dropout(dropout_rate)
        # 入力層
        prev_size = self.input_size
        for layer_size in self.layer_size:
            self.layers.append(nn.Linear(prev_size, layer_size))
            prev_size = layer_size
        # 出力層
        self.out = nn.Linear(prev_size, self.output_size)
        if output_activation == "sig":
            self.activation_out = nn.Sigmoid()
        elif output_activation == "softmax":
            self.activation_out = nn.Softmax(dim=1)  # 正しい構文
        else:
            raise ValueError("Unsupported output activation. Use 'sig' or 'softmax'.")
        self.activation = activation

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activations[self.activation](x)
            x = self.dropout(x)
        x = self.out(x)
        x = self.activation_out(x)
        return x
```

今回はニューラルネットワークを使用しました。引数でニューロンの数や活性化関数を指定できるようにしています。活性化関数にはReLUが良く使われますが、それよりも精度が高いと言われているMish関数を使用しました。また、出力層の活性化関数には３着以内に入る確率を出すのでシグモイド関数を使用しました。

## 3.データローダの作成

次に用意したデータセットを読み込むためのデータローダを作成します。  

```python
def make_data_loader(dataset, ratio: [int, int, int], batch=32, ensemble=0):
    """データローダーを作る関数

    Args:
        dataset (_type_): 使用するデータセット
        ratio (int, int, int]): 訓練、検証、テストデータの比率
        batch (int, optional): バッチサイズ。デフォルトは32
        ensemble (int, optional): 訓練データの分割数（アンサンブル学習用）。デフォルトは0(分けない)

    Returns:
        _type_: _description_
    """
    dataset_size = len(dataset)
    train_rate = ratio[0] / sum(ratio)
    val_rate = ratio[1] / sum(ratio)
    # 分割比率を設定 (例: 訓練:検証:テスト = 70%:15%:15%)
    train_size = int(dataset_size * train_rate)
    val_size = int(dataset_size * val_rate)
    test_size = dataset_size - train_size - val_size  # 残りをテストとする

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    if ensemble:
        sub_train_sizes = [
            int(train_size / ensemble) for _ in range(ensemble)
        ]  # 最初の4つのサイズ
        sub_train_sizes.append(
            train_size - sum(sub_train_sizes)
        )  # 最後のサブセットのサイズ
        # 分割された訓練データセットのインデックスを生成
        indices = torch.randperm(train_size).tolist()
        train_dataset = [
            Subset(
                train_dataset,
                indices[sum(sub_train_sizes[:i]) : sum(sub_train_sizes[: i + 1])],
            )
            for i in range(ensemble)
        ]
        # 各サブセットに対応するDataLoaderを作成
        train_loader = [
            DataLoader(dataset, batch_size=batch, shuffle=True, pin_memory=True)
            for dataset in train_dataset
        ]
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch,
            shuffle=False,
            pin_memory=True,
        )
    # 検証、確認用のDataLoaderを作成
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch,
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
```

今回の学習では訓練データ、検証データ、テストデータを70:15:15の比率でそれぞれ分割しました。アンサンブル学習用の引数も用意してますが、最終的なモデルの作成では使っていません。

## 4.学習

### 4.1 独自誤差関数

競馬のレースは最大18頭ですが、全レースで18頭全てが出走するわけではなく少ないときは５頭程度の時もあります。なので馬番ごとの成績にも偏りが生じています。  
それを考慮するために誤差関数を新たに作成します。  

```python
class WeightedBCELossMulti(nn.Module):
    def __init__(self):
        super(WeightedBCELossMulti, self).__init__()
        self.weight_tensor_one = torch.tensor(
            [
                4.0467,
                3.8973,
                4.0508,
                3.7718,
                4.0142,
                3.8689,
                4.0264,
                4.0231,
                4.4075,
                4.6175,
                5.3236,
                5.6824,
                7.1395,
                8.0703,
                9.9560,
                12.4485,
                85.8520,
                96.1656,
            ],
            dtype=torch.float,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, logits, targets):
        # ラベルが1の要素に対してweight_tensor_oneの重みを適用し、
        # ラベルが0の要素に対しては1の重みを適用
        weights = torch.where(
            targets == 1, self.weight_tensor_one, torch.ones_like(targets)
        )
        weights = weights.to(targets.device)  # weightsをtargetsと同じデバイスに移動
        # 重み付きバイナリクロスエントロピー損失の計算
        loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
        return loss


class WeightedBCELossSingle(nn.Module):
    def __init__(self):
        super(WeightedBCELossSingle, self).__init__()
        self.weight_tensor_one = torch.tensor(
            [
                14.4231,
                13.7151,
                14.2716,
                13.3220,
                14.3394,
                13.4956,
                13.9918,
                14.0644,
                15.4426,
                15.6629,
                17.5519,
                18.9955,
                23.4058,
                25.5072,
                31.7985,
                38.7101,
                271.7456,
                264.7521,
            ],
            dtype=torch.float,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, logits, targets):
        # ラベルが1の要素に対してweight_tensor_oneの重みを適用し、
        # ラベルが0の要素に対しては1の重みを適用
        weights = torch.where(
            targets == 1, self.weight_tensor_one, torch.ones_like(targets)
        )
        weights = weights.to(targets.device)  # weightsをtargetsと同じデバイスに移動
        # 重み付きバイナリクロスエントロピー損失の計算
        loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
        return loss
```

各重みはあらかじめ全レースで馬番ごとの成績を集計し、  
複勝では4着以下の数/3着以内の数の値  
単勝では2着以下の数/1着の数の値  
を使用しました。

### 4.2 モデルの作成

```python
model_multi = NNClassifier(1534, 18, [128, 128]).to(device)
model_single = NNClassifier(1534, 18, [128, 128]).to(device)

lr = 1e-6
optimizer_multi = torch.optim.AdamW(model_multi.parameters(), lr=lr)
optimizer_single = torch.optim.AdamW(model_single.parameters(), lr=lr)
criterion_multi = WeightedBCELossMulti().to(device)
criterion_single = WeightedBCELossSingle().to(device)
```

入力1534、出力18、ニューロン数128*2層の隠れ層を持つモデルにしました。  
また、最適化にはadamwを使用しました。  
複勝で学習させるモデルと単勝で学習させるモデルを別で作成していますが使用する学習データ以外は特に違いはありません。

### 4.3 学習

```python
# 誤差、ラベル1の正答率、ラベル0の正答率を記録するリストを初期化
losses = []
accuracies_1 = []
accuracies_0 = []
# 前回との正答率の差分
diff_acc1 = 0
# 前回の正答率
acc1_tmp = -1

for epoch in range(epochs):
    epoch_losses = []
    for batch_idx, (data, targets) in enumerate(train_loader_multi):
        x = data.to(device)
        y = targets.to(device)
        scores = model_multi(x)
        loss = criterion_multi(scores, y)
        optimizer_multi.zero_grad()
        loss.backward()
        optimizer_multi.step()
        epoch_losses.append(loss.item())
    # エポックの平均誤差を記録
    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(epoch_loss)

    # 正答率を計算して記録
    accuracy_1, accuracy_0 = check_accuracy_topk(val_loader_multi, model_multi, 3)
    accuracies_1.append(accuracy_1)
    accuracies_0.append(accuracy_0)

    if (epoch + 1) % show_num == 0:
        print(
            f"Epoch {epoch + 1:.4g}: Loss = {epoch_loss:.5f}, Accuracy 1 = {accuracy_1:.2f}%, Accuracy 0 = {accuracy_0:.2f}%"
        )

    if (epoch + 1) % 50 == 0:  # 50エポックごとに確認する
        # ２回連続で正答率が下がったら学習を強制終了する
        if diff_acc1 < 0 and (accuracy_1 - acc1_tmp) < 0:
            print(f"epoch{epoch}, diff{diff_acc1}, {accuracy_1 - acc1_tmp}")
            print("Force close because the accuracy has decreased twice in a row")
            break
        # 現在の正答率とその変化量を更新
        diff_acc1 = accuracy_1 - acc1_tmp if epoch > 0 else 0
        acc1_tmp = accuracy_1


# モデルのパラメータ
model_params = {
    "input_size": model_multi.input_size,
    "output_size": model_multi.output_size,
    "layers": model_multi.layer_size,  # layers_sizesは各隠れ層のノード数をリストで保持する属性です
    "dropout_rate": model_multi.dropout_rate,  # dropout_rateはドロップアウト率を保持する属性です
    "activation": model_multi.activation,  # activation_typeは活性化関数の種類を保持する属性です
    "output_activation": model_multi.output_activation,  # output_activation_typeは出力層の活性化関数の種類を保持する属性です
}

# モデルの状態とパラメータを辞書に格納
model_info = {"state_dict": model_multi.state_dict(), "params": model_params}

# モデル情報をファイルに保存
torch.save(model_info, f"../models/nn/multi-test.pth")

# 学習が完了した後、誤差と正答率の推移をプロット
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(losses, label="Loss")
plt.title("Loss during training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(accuracies_1, label="Accuracy for label 1")
plt.title("Accuracy for label 1 during training")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(accuracies_0, label="Accuracy for label 0")
plt.title("Accuracy for label 0 during training")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
```

内容的にほとんど同じなので複勝モデルの学習のみ記載しています。  
誤差、ラベル1に対する正答率、ラベル0に対する正答率を記録しています。  
50エポックごとに正答率を確認し、2回連続で正答率が下がったら学習を強制終了します。  
学習が終わった後は誤差と正答率の推移をプロットしています。

## 5.モデルの評価

```python
def check_accuracy_topk_test(loaders, model_multi, model_single, k=3):
    num_correct_1 = 0
    num_samples_1 = 0
    num_correct_0 = 0
    num_samples_0 = 0
    model_multi.eval()
    model_single.eval()
    with torch.no_grad():
        for x, y in loaders:
            x = x.to(device)
            y = y.to(device)
            scores_multi = model_multi(x)
            scores_single = model_single(x)
            # 予測の平均を計算
            avg_predictions = torch.mean(
                torch.stack([scores_multi, scores_single]), dim=0
            )
            # 上位k個の予測を1に、それ以外を0にする
            topk_predictions = torch.zeros_like(avg_predictions, device=device)
            topk_vals, topk_indices = avg_predictions.topk(k, dim=1)
            topk_predictions.scatter_(1, topk_indices, 1)
            # 1の場合
            correct_predictions_1 = topk_predictions.bool() & y.bool()
            num_correct_1 += correct_predictions_1.type(torch.float).sum().item()
            num_samples_1 += y.sum().item()
            # 0の場合
            correct_predictions_0 = (~topk_predictions.bool()) & (~y.bool())
            num_correct_0 += correct_predictions_0.type(torch.float).sum().item()
            num_samples_0 += (1 - y).sum().item()

    # 正解率の計算
    accuracy_1 = (num_correct_1 / num_samples_1 * 100) if num_samples_1 > 0 else 0
    accuracy_0 = (num_correct_0 / num_samples_0 * 100) if num_samples_0 > 0 else 0
    model_multi.train()
    model_single.train()

    return f"複合モデル - ラベル1の正答率: {accuracy_1:.2f}%, ラベル0の正答率: {accuracy_0:.2f}%"


def check_accuracy_topk_test_compare(loaders, model_multi, model_single, k=3):
    num_correct_1_multi = 0
    num_samples_1_multi = 0
    num_correct_0_multi = 0
    num_samples_0_multi = 0
    num_correct_1_single = 0
    num_samples_1_single = 0
    num_correct_0_single = 0
    num_samples_0_single = 0
    model_multi.eval()
    model_single.eval()

    with torch.no_grad():
        for x, y in loaders:
            x = x.to(device)
            y = y.to(device)
            scores_multi = model_multi(x)
            scores_single = model_single(x)
            # 複勝モデルの正答率計算
            topk_predictions_multi = torch.zeros_like(scores_multi, device=device)
            topk_vals_multi, topk_indices_multi = scores_multi.topk(k, dim=1)
            topk_predictions_multi.scatter_(1, topk_indices_multi, 1)
            correct_predictions_1_multi = topk_predictions_multi.bool() & y.bool()
            num_correct_1_multi += (
                correct_predictions_1_multi.type(torch.float).sum().item()
            )
            num_samples_1_multi += y.sum().item()
            correct_predictions_0_multi = (~topk_predictions_multi.bool()) & (~y.bool())
            num_correct_0_multi += (
                correct_predictions_0_multi.type(torch.float).sum().item()
            )
            num_samples_0_multi += (1 - y).sum().item()
            # 単勝モデルの正答率計算
            topk_predictions_single = torch.zeros_like(scores_single, device=device)
            topk_vals_single, topk_indices_single = scores_single.topk(k, dim=1)
            topk_predictions_single.scatter_(1, topk_indices_single, 1)
            correct_predictions_1_single = topk_predictions_single.bool() & y.bool()
            num_correct_1_single += (
                correct_predictions_1_single.type(torch.float).sum().item()
            )
            num_samples_1_single += y.sum().item()
            correct_predictions_0_single = (~topk_predictions_single.bool()) & (
                ~y.bool()
            )
            num_correct_0_single += (
                correct_predictions_0_single.type(torch.float).sum().item()
            )
            num_samples_0_single += (1 - y).sum().item()

    accuracy_1_multi = (
        (num_correct_1_multi / num_samples_1_multi * 100)
        if num_samples_1_multi > 0
        else 0
    )
    accuracy_0_multi = (
        (num_correct_0_multi / num_samples_0_multi * 100)
        if num_samples_0_multi > 0
        else 0
    )
    accuracy_1_single = (
        (num_correct_1_single / num_samples_1_single * 100)
        if num_samples_1_single > 0
        else 0
    )
    accuracy_0_single = (
        (num_correct_0_single / num_samples_0_single * 100)
        if num_samples_0_single > 0
        else 0
    )
    model_multi.train()
    model_single.train()

    return (
        f"複勝モデル - ラベル1の正答率: {accuracy_1_multi:.2f}%, ラベル0の正答率: {accuracy_0_multi:.2f}%\n"
        f"単勝モデル - ラベル1の正答率: {accuracy_1_single:.2f}%, ラベル0の正答率: {accuracy_0_single:.2f}%"
    )

def model_load(path: str):
    # モデル情報をファイルから読み込む
    model_info = torch.load(path)

    # 読み込んだパラメータを使用してモデルインスタンスを作成
    model_params = model_info["params"]
    model_reconstructed = NNClassifier(
        input_size=model_params["input_size"],
        output_size=model_params["output_size"],
        layers=model_params["layers"],
        dropout_rate=model_params["dropout_rate"],
        activation=model_params["activation"],
        output_activation=model_params["output_activation"],
    )

    # モデルの状態を読み込んだ情報で更新
    model_reconstructed.load_state_dict(model_info["state_dict"])

    # 必要に応じてモデルを適切なデバイスに移動
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model_reconstructed.to(device)


model_multi_t = model_load(f"../models/nn/multi-pred.pth")
model_single_t = model_load(f"../models/nn/single-pred.pth")

print("---k=3---")
print(check_accuracy_topk_test(test_loader_multi, model_multi_t, model_single_t, 3))
print(
    check_accuracy_topk_test_compare(
        test_loader_multi, model_multi_t, model_single_t, 3
    )
)
print("---k=5---")
print(check_accuracy_topk_test(test_loader_multi, model_multi_t, model_single_t, 5))
print(
    check_accuracy_topk_test_compare(
        test_loader_multi, model_multi_t, model_single_t, 5
    )
)
```

複勝モデル、単勝モデルのそれぞれで  
推論した確率が最も高いもの3個もしくは５個を1としたときの正答率を出しています。
