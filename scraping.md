# 学習データを作成するための元データの取得

今回は

- 過去のレース結果
- 馬の過去成績
- 親馬の成績

の３種類のデータを使用しました。データは[netkeiba](https://www.netkeiba.com/)さんのデータベースをスクレイピングして取得しました。  

## 1.過去のレース結果

### 1-1.クラス

以下はレース結果のスクレイピング処理をまとめたクラスです。

```python
class RaceResult:
    def __init__(self, url: str) -> None:
        self.url = url
        self.race_result = self.fetch_race_results(self.url)

    def fetch_race_results(self, url: str) -> pd.DataFrame:
        """レース結果を取得する

        Args:
            url (str): レース結果のURL

        Raises:
            ValueError: レース結果の取得できなかった場合はエラーを返す

        Returns:
            pd.DataFrame: レース結果
        """
        # 準備＆結果取得
        response = requests.get(url)
        response.encoding = "EUC-JP"
        html_string = io.StringIO(response.text)
        results = pd.read_html(html_string)[0]
        soup = bs(html_string, "html.parser")
        # 馬のID
        horse_id_list = []
        horse_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
            "a", attrs={"href": re.compile("^/horse")}
        )
        for a in horse_a_list:
            horse_id = re.findall(r"[0-9]+", a["href"])
            horse_id_list.append(horse_id[0])
        # 騎手のID
        jockey_id_list = []
        jockey_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
            "a", attrs={"href": re.compile("^/jockey")}
        )
        for a in jockey_a_list:
            jockey_id = re.findall(r"[0-9]+", a["href"])
            jockey_id_list.append(jockey_id[0])

        results["horse_id"] = horse_id_list
        results["jockey_id"] = jockey_id_list
        # データ整形
        results.columns = results.columns.str.replace(" ", "")
        results = results.sort_values(by="馬番")
        results = results.reset_index(drop=True)

        # レース情報
        lxml_data = html.fromstring(str(soup))
        raw_info1 = lxml_data.xpath(
            "//*[@id='main']/div/div/div/diary_snap/div/div/dl/dd/p/diary_snap_cut/span"
        )[0]
        raw_info_text1 = re.sub(r"\s", "", raw_info1.text)
        for item in raw_info_text1.split("/"):
            # 障害レースの時の距離と向き
            if re.match(r"(障.*)[0-9]{,4}m", item):
                results["course_type"] = "障"
                results["course_way"] = "無"
                results["course_length"] = re.search(r"[0-9]{0,4}m", item).group()[:-1]
            # 芝かダートの時の距離と向き
            elif re.match(r"(芝|ダ)(右|左|直線).*[0-9]{,4}m", item):
                item_replace = item.replace(" ", "").replace("直線", "直")
                results["course_type"] = item_replace[0]
                results["course_way"] = item_replace[1]
                results["course_length"] = re.search(
                    r"[0-9]{0,4}m", item_replace
                ).group()[:-1]
            # 天候取得
            elif "天候:" in item:
                results["weather"] = item[-1]
            # 馬場状態の取得
            elif "芝:" in item or "ダート:" in item:
                if any(results["course_type"] == "障"):
                    results["state_grass"] = item[2]
                    results["state_dirt"] = item[-1]
                elif any(results["course_type"] == "芝"):
                    results["state_grass"] = item[-1]
                    results["state_dirt"] = "無"
                elif any(results["course_type"] == "ダ"):
                    results["state_grass"] = "無"
                    results["state_dirt"] = item[-1]
        raw_info2 = lxml_data.xpath("//*[@id='main']/div/div/div/diary_snap/div/div/p")[
            0
        ]
        raw_info_text2 = raw_info2.text
        for item in raw_info_text2.split(" "):
            item = (
                item.replace("500万下", "1勝")
                .replace("1000万下", "2勝")
                .replace("1600万下", "3勝")
            )
            # レースのクラス
            match_race_class = re.search(r"新馬|未勝利|1勝|2勝|3勝|オープン", item)
            if match_race_class:
                results["class"] = match_race_class.group()
            # レースの日付
            elif re.match(r"[0-9]{4}年[0-9]{,2}月[0-9]{,2}日", item):
                results["date"] = item
            # レースの開催場所
            elif re.match(r"[0-9]*回.*[0-9]*日目", item):
                text = re.sub(r"[0-9]*回", "", item)
                text = re.sub(r"[0-9]*日目", "", text)
                results["place"] = text
        race_name = lxml_data.xpath(
            '//*[@id="main"]/div/div/div/diary_snap/div/div/dl/dd/h1/text()'
        )[0]
        match_race_class = re.search(r"G1|G2|G3|L", race_name)
        if match_race_class:
            results["class"] = match_race_class.group()
        if not ("class" in results.columns):
            print(raw_info_text1.split("/"))
            print(raw_info_text2.split(" "))
            print(race_name)
            raise ValueError("row of class is not exist")
        # ラウンド
        race_round = lxml_data.xpath(
            '//*[@id="main"]/div/div/div/diary_snap/div/div/dl/dt'
        )[0].text
        race_round = re.sub(r"[R\s\n]", "", race_round)
        results["round"] = race_round
        return results

    def save_race_results_pkl(self, path: str) -> None:
        """レース結果をpklで保存する

        Args:
            path (str): 保存するパス
        """
        self.race_result.to_pickle(path)

    def save_race_results_csv(self, path: str) -> None:
        """レース結果をcsvで保存する

        Args:
            path (str): 保存するパス
        """
        self.race_result.to_csv(path)
```

### 1-2.説明

レース結果はtable要素で実装されているため、`results = pd.read_html(html_string)[0]`でデータフレームとして取得します。  
馬の過去成績をスクレイピングするために必要な馬のIDもこのタイミングで取得しておきます。

```python
# 馬のID
horse_id_list = []
horse_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
    "a", attrs={"href": re.compile("^/horse")}
)
for a in horse_a_list:
    horse_id = re.findall(r"[0-9]+", a["href"])
    horse_id_list.append(horse_id[0])
# 騎手のID
jockey_id_list = []
jockey_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
    "a", attrs={"href": re.compile("^/jockey")}
)
for a in jockey_a_list:
    jockey_id = re.findall(r"[0-9]+", a["href"])
    jockey_id_list.append(jockey_id[0])
```

馬場や天気といったその他の情報についてはテキストであるため、正規表現などを使ってテキストから必要な情報を取得しています。

```python
lxml_data = html.fromstring(str(soup))
raw_info1 = lxml_data.xpath(
    "//*[@id='main']/div/div/div/diary_snap/div/div/dl/dd/p/diary_snap_cut/span"
)[0]
raw_info_text1 = re.sub(r"\s", "", raw_info1.text)
for item in raw_info_text1.split("/"):
    # 障害レースの時の距離と向き
    if re.match(r"(障.*)[0-9]{,4}m", item):
        results["course_type"] = "障"
        results["course_way"] = "無"
        results["course_length"] = re.search(r"[0-9]{0,4}m", item).group()[:-1]
    # 芝かダートの時の距離と向き
    elif re.match(r"(芝|ダ)(右|左|直線).*[0-9]{,4}m", item):
        item_replace = item.replace(" ", "").replace("直線", "直")
        results["course_type"] = item_replace[0]
        results["course_way"] = item_replace[1]
        results["course_length"] = re.search(
            r"[0-9]{0,4}m", item_replace
        ).group()[:-1]
    # 天候取得
    elif "天候:" in item:
        results["weather"] = item[-1]
    # 馬場状態の取得
    elif "芝:" in item or "ダート:" in item:
        if any(results["course_type"] == "障"):
            results["state_grass"] = item[2]
            results["state_dirt"] = item[-1]
        elif any(results["course_type"] == "芝"):
            results["state_grass"] = item[-1]
            results["state_dirt"] = "無"
        elif any(results["course_type"] == "ダ"):
            results["state_grass"] = "無"
            results["state_dirt"] = item[-1]
raw_info2 = lxml_data.xpath("//*[@id='main']/div/div/div/diary_snap/div/div/p")[
    0
]
raw_info_text2 = raw_info2.text
for item in raw_info_text2.split(" "):
    item = (
        item.replace("500万下", "1勝")
        .replace("1000万下", "2勝")
        .replace("1600万下", "3勝")
    )
    # レースのクラス
    match_race_class = re.search(r"新馬|未勝利|1勝|2勝|3勝|オープン", item)
    if match_race_class:
        results["class"] = match_race_class.group()
    # レースの日付
    elif re.match(r"[0-9]{4}年[0-9]{,2}月[0-9]{,2}日", item):
        results["date"] = item
    # レースの開催場所
    elif re.match(r"[0-9]*回.*[0-9]*日目", item):
        text = re.sub(r"[0-9]*回", "", item)
        text = re.sub(r"[0-9]*日目", "", text)
        results["place"] = text
race_name = lxml_data.xpath(
    '//*[@id="main"]/div/div/div/diary_snap/div/div/dl/dd/h1/text()'
)[0]
match_race_class = re.search(r"G1|G2|G3|L", race_name)
if match_race_class:
    results["class"] = match_race_class.group()
if not ("class" in results.columns):
    print(raw_info_text1.split("/"))
    print(raw_info_text2.split(" "))
    print(race_name)
    raise ValueError("row of class is not exist")
# ラウンド
race_round = lxml_data.xpath(
    '//*[@id="main"]/div/div/div/diary_snap/div/div/dl/dt'
)[0].text
race_round = re.sub(r"[R\s\n]", "", race_round)
results["round"] = race_round
```

2019年にレースのクラスの名称が変わっていますが、今回は現在の名称に統一しています。

```python
item = (
    item.replace("500万下", "1勝")
    .replace("1000万下", "2勝")
    .replace("1600万下", "3勝")
)
```

netkeibaさんのレースにはIDが割り振られています。  
西暦(４桁)＋レース場ID(1~10)＋開催回数＋開催日数＋ラウンド数  
例）2023年天皇賞秋→東京第4回開催の9日目11ラウンド→202305040911  
  
これを利用してレースIDのリストを以下で作成できます。  
年については1年分でいいなら文字列の頭に西暦を入れ、2年分以上とる場合はfor文をもう1個入れるなりしてください。

```python
for place in range(1, 11, 1):
    for kai in range(1, 7, 1):
        for day in range(1, 13, 1):
            for r in range(1, 13, 1):
                race_id = (
                    str(place).zfill(2)
                    + str(kai).zfill(2)
                    + str(day).zfill(2)
                    + str(r).zfill(2)
                )
```

あとは作成したリストに対してfor文でスクレイピング処理を回してデータを取得・保存します。

```python
for i in tqdm(race_id_list):
    try:
        scraping = RaceResult(f"https://db.netkeiba.com/race/2022{i}/")
        #pickle形式で保存
        scraping.save_race_results_pkl(f"../Raw-Data/Race-Results/2022/{i}.pkl")
        time.sleep(1)
    except IndexError as e:
        time.sleep(1)
        skip_race.append(f"{i}")
        continue
    except Exception as e:
        print(i)
        raise e
```

[データサンプル](./Raw-Data/sample/Race-Results.csv)

短時間で大量のリクエストを送るとサーバーに負荷をかけてしまうので`time.sleep(1)`で1秒で1ページのスクレイピングを行うようにしています。
> ["!note]
> 1秒待つというルールがありますが、1経験的な値であるため必ずしも1秒待てば良いということではありません
> [!WARNING]
> 法律は詳しくないですが、大量のリクエストを送ってしましサーバーに高負荷を掛けてしまった場合、警察のお世話になるかもしれないので、何秒か待つ処理は絶対に入れましょう

## 2.馬の過去成績

### 2-1.クラス

```python
class HorseResult:
    def __init__(self, horse_path: str, save_path: str) -> None:
        self.race_result = pd.read_pickle(horse_path)
        self.horse_ids = self.race_result["horse_id"]
        self.save_path = save_path

    def scraping(self, horse_id: str) -> pd.DataFrame | None:
        """レース結果を取得する

        Args:
            horse_id (str): 騎手のID

        Raises:
            e: レース結果の取得できなかった場合はエラーを返す

        Returns:
            pd.DataFrame | None: レース結果
        """
        try:
            url = f"https://db.netkeiba.com/horse/{horse_id}"
            response = requests.get(url)
            response.encoding = "EUC-JP"
            html_string = io.StringIO(response.text)
            df = pd.read_html(html_string)[3]
            if df.columns[0] == "受賞歴":
                df = pd.read_html(html_string)[4]
            time.sleep(1)
            return df
        except Exception as e:
            print(horse_id)
            raise e

    def save(self, data: pd.DataFrame, name: str) -> None:
        """レース結果を保存する

        Args:
            data (pd.DataFrame): レース結果
            name (str): 保存するファイル名
        """
        if data is None:
            return
        df: pd.DataFrame = data
        df.to_pickle(f"{self.save_path}/{name}.pkl")

    def scrape_save(self) -> None:
        """馬の過去成績を取得して保存する"""
        for horse_id in self.horse_ids:
            if os.path.isfile(f"{self.save_path}/{horse_id}.pkl"):
                continue
            df = self.scraping(horse_id)
            self.save(df, horse_id)
```

### 2-2.概要

過去成績のURLには馬のIDが使われているため、レース結果のスクレイピングで取得しておいた馬のIDを使ってスクレイピングを行います。  
基本的にはレース結果と同じくtable要素で作られているためデータフレームで取得しますが、過去に年度代表馬等に選出されている場合は取得するtable要素が一つずれます。

```python
url = f"https://db.netkeiba.com/horse/{horse_id}"
response = requests.get(url)
response.encoding = "EUC-JP"
html_string = io.StringIO(response.text)
df = pd.read_html(html_string)[3]
if df.columns[0] == "受賞歴":
    df = pd.read_html(html_string)[4]
```

肝心の馬のIDはレース結果のデータの中に格納されているので、保存しているレース結果のデータから馬のIDを取得しています。

```python
# file_list : レース結果データのファイルパスのリスト
# save_path : 馬の過去成績の保存先パス
for filename in tqdm(file_list):
    else:
        horse_results = HorseResult(f"{filename}", save_path=save_path)
        horse_results.scrape_save()
```

[データサンプル](./Raw-Data/sample/Horse-Results.csv)

## 3.馬の血統

馬の血統については直接学習データに組み込むわけではないですが、馬とその親馬を調べるために使用しています。

### 3-1.クラス

```python
class PedigreeInfo:
    def __init__(self, horse_path: str, save_path: str) -> None:
        self.race_result = pd.read_pickle(horse_path)
        self.horse_ids = self.race_result["horse_id"]
        self.save_path = save_path

    def scrape(self, horse_id: str) -> list[str] | None:
        """血統情報を取得する

        Args:
            horse_id (str): 馬のID

        Raises:
            e: 血統情報の取得ができなかった場合はエラーをそのまま返す

        Returns:
            list[str] | None: 血統情報
        """
        try:
            url = f"https://db.netkeiba.com/horse/{horse_id}"
            response = requests.get(url)
            response.encoding = "EUC-JP"
            html_string = io.StringIO(response.text)
            soup = bs(html_string, "html.parser")
            peds = []
            id_list = soup.find("table", attrs={"class": "blood_table"}).find_all("a")
            for i in id_list[0], id_list[4]:
                peds.append(i.get("href").replace("/horse/ped/", "").replace("/", ""))
            peds
            time.sleep(1)
            return peds
        except Exception as e:
            print(horse_id)
            raise e

    def save(self, data: list[str], name: str) -> None:
        """血統情報を保存する

        Args:
            data (list[str]): 血統情報
            name (str): 保存するファイル名
        """
        if data is None:
            return
        with open(f"{self.save_path}/{name}.pickle", "wb") as f:
            pickle.dump(data, f)

    def scrape_save(self) -> None:
        for horse_id in self.horse_ids:
            if os.path.isfile(f"{self.save_path}/{horse_id}.pickle"):
                continue
            peds_list = self.scrape(horse_id)
            self.save(peds_list, horse_id)
```

### 3-2.概要

馬の過去成績ページに血統情報もあるため、そこから父親と母方の祖父のIDを取得しています。  
外国産馬などでは血統情報がない場合があります。その場合は何も保存しません。

馬の過去成績と同様レース結果のデータを元にスクレイピングしています。

```python
for filename in tqdm(file_list):
    else:
        pedigree = PedigreeInfo(f"{filename}", save_path=save_path)
        pedigree.scrape_save()
```

[データサンプル](./Raw-Data/sample/Pedigree.csv)

## 4.親馬の過去成績

### 4-1.クラス

```python
class PedigreeResults:
    columns = [
        "日付",
        "開催",
        "天 気",
        "R",
        "レース名",
        "映 像",
        "頭 数",
        "枠 番",
        "馬 番",
        "オ ッ ズ",
        "人 気",
        "着 順",
        "騎手",
        "斤 量",
        "距離",
        "馬 場",
        "馬場 指数",
        "タイム",
        "着差",
        "ﾀｲﾑ 指数",
        "通過",
        "ペース",
        "上り",
        "馬体重",
        "厩舎 ｺﾒﾝﾄ",
        "備考",
        "勝ち馬 (2着馬)",
        "賞金",
    ]

    def __init__(self, horse_path: str, save_path: str) -> None:
        with open(horse_path, "rb") as f:
            self.horse_ids = pickle.load(f)
        self.save_path = save_path

    def scraping(self, horse_id: str) -> pd.DataFrame | None:
        """親馬の過去成績を取得する

        Args:
            horse_id (str): 親馬のID

        Raises:
            e: 親馬の過去成績の取得ができなかった場合はエラーをそのまま返す

        Returns:
            pd.DataFrame | None: 親馬の過去成績
        """
        try:
            url = f"https://db.netkeiba.com/horse/{horse_id}"
            response = requests.get(url)
            response.encoding = "EUC-JP"
            html_string = io.StringIO(response.text)
            df = pd.read_html(html_string)[3]
            # 年度代表馬のテーブルだったら取り直す
            if df.columns[0] != "日付":
                time.sleep(1)
                df = pd.read_html(html_string)[4]
                # それでも過去成績が取れなければ0埋めのデータとする
                if df.columns[0] != "日付":
                    df = pd.DataFrame(0, index=range(1), columns=self.columns)
            time.sleep(1)
            return df
        except Exception as e:
            print(horse_id)
            raise e

    def save(self, data: pd.DataFrame, name: str) -> None:
        """親馬の過去成績を保存する

        Args:
            data (pd.DataFrame): レース結果
            name (str): 保存するファイル名
        """
        if data is None:
            return
        df: pd.DataFrame = data
        df.to_pickle(f"{self.save_path}/{name}.pkl")

    def scrape_save(self) -> None:
        for horse_id in self.horse_ids:
            if os.path.isfile(f"{self.save_path}/{horse_id}.pkl"):
                continue
            df = self.scraping(horse_id)
            self.save(df, horse_id)
```

### 4-2.概要

馬の過去成績と同じ要領で取得しています。  
親馬の場合は過去成績がないケースもあるため、年度代表馬のテーブルだったら取り直す処理を行ったうえでデータが取得できなかった場合は保存しません。

スクレイピングの際は3で保存した馬の血統情報を元にスクレイピングしています。

```python
#dir_list : 馬の血統情報のファイルパスのリスト
for filename in tqdm(dir_list):
    try:
        pedigree_results = PedigreeResults(
            f"{dir_path}/{filename}", save_path=save_path
        )
        pedigree_results.scrape_save()
    except Exception as e:
        print(filename)
        raise e
```

[データサンプル](./Raw-Data/sample/Pedigree-Results.csv)
