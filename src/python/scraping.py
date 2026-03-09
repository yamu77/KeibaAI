"""競馬データのスクレイピングを行うモジュール。

このモジュールは、netkeiba.comからレース結果、競走馬の過去成績、
血統情報、騎手情報などの競馬データを取得するためのクラスを提供します。
"""

import pickle
import time
import os
import io
import re
from abc import ABC, abstractmethod
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from lxml import html

# ============================================================================
# 基底クラス
# ============================================================================


class BaseScraper(ABC):
    """スクレイピング用の基底クラス

    共通のHTTPヘッダーを提供します。
    """

    def __init__(self):
        """共通のHTTPヘッダーを初期化"""
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/130.0.0.0 Safari/537.36"
            )
        }

    @abstractmethod
    @classmethod
    def save(cls, data, path: str, **kwargs) -> None:
        """データを保存する抽象メソッド

        Args:
            data: 保存するデータ
            path (str): 保存先のファイルパス
            **kwargs: その他のオプション引数
        """


# ============================================================================
# ユーティリティ関数
# ============================================================================


def get_all_files_in_directory(path: str) -> list[str]:
    """
    指定したパスの配下にあるすべてのファイル（子孫ファイルも含む）のリストを返します。

    :param path: ファイルを検索するディレクトリのパス
    :return: ファイルのパスのリスト
    """
    files_list = []
    for root, _, files in os.walk(path):
        for file_name in files:
            files_list.append(f"{root}/{file_name}")
    return files_list


def write_path(path: str, write_list: list[str]) -> None:
    """パスを書き込む

    Args:
        path (str): 書き込むパス
        write_list (list[str]): 書き込むリスト
    """
    with open(path, "a", encoding="utf-8") as f:
        for i in write_list:
            f.write(f"{i}\n")


def read_file(path: str) -> list[str]:
    """ファイルを読み込む

    Args:
        path (str): 読み込むパス

    Returns:
        list[str]: 読み込んだリスト
    """
    with open(path, "r", encoding="utf-8") as f:
        read_list = f.readlines()
    new_list = []
    for i in read_list:
        new_list.append(i.replace("\n", ""))
    return new_list


class RaceResult(BaseScraper):
    """レース結果のスクレイピング"""

    def __init__(self, url: str) -> None:
        super().__init__()
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
        response = requests.get(url, headers=self.headers)
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
            "//*[@id='main']/div/div/div/diary_snap"
            "/div/div/dl/dd/p/diary_snap_cut/span"
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
        if "class" not in results.columns:
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

    @classmethod
    def save(
        cls, data: pd.DataFrame, path: str, file_format: str = "pkl", **kwargs
    ) -> None:
        """レース結果を保存する

        Args:
            data (pd.DataFrame): 保存するレース結果データ
            path (str): 保存するパス
            file_format (str): 保存形式（'pkl'または'csv'）。デフォルトは'pkl'
            **kwargs: その他のオプション引数
        """
        if file_format == "pkl":
            data.to_pickle(path)
        elif file_format == "csv":
            data.to_csv(path)
        else:
            raise ValueError(f"Unsupported format: {file_format}. Use 'pkl' or 'csv'")


class HorseResult(BaseScraper):
    """競走馬の過去成績のスクレイピング"""

    def __init__(self, horse_path: str, save_path: str) -> None:
        super().__init__()
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
            response = requests.get(url, headers=self.headers)
            response.encoding = "EUC-JP"
            html_string = io.StringIO(response.text)
            dfs = pd.read_html(html_string)
            for df in dfs:
                if df.columns[0] == "日付":
                    return df
            return None
        except Exception as e:
            print(horse_id)
            raise e

    @classmethod
    def save(cls, data: pd.DataFrame, path: str, **kwargs) -> None:
        """レース結果を保存する

        Args:
            data (pd.DataFrame): レース結果
            path (str): 保存するファイルパス（拡張子を含む）
            **kwargs: その他のオプション引数
        """
        if data is None:
            return
        data.columns = data.columns.astype(str)
        data.columns = data.columns.str.replace(" ", "")
        data = data.sort_values(by="馬番")
        data = data.reset_index(drop=True)
        data.to_pickle(path)

    def scraping_and_save(self) -> None:
        """レース結果を取得して保存する"""
        for horse_id in self.horse_ids:
            file_path = f"{self.save_path}/{horse_id}.pkl"
            if os.path.isfile(file_path):
                continue
            df = self.scraping(horse_id)
            self.save(df, file_path)
            time.sleep(1)


class PedigreeInfo(BaseScraper):
    """馬の血統情報のスクレイピング"""

    def __init__(self, horse_path: str, save_path: str) -> None:
        super().__init__()
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
            response = requests.get(url, headers=self.headers)
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

    @classmethod
    def save(cls, data: list[str] | None, path: str, **kwargs) -> None:
        """血統情報を保存する

        Args:
            data (list[str] | None): 血統情報
            path (str): 保存するファイルパス（拡張子を含む）
            **kwargs: その他のオプション引数
        """
        if data is None:
            return
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def scrape_save(self) -> None:
        for horse_id in self.horse_ids:
            # 既にファイルが存在しているなら飛ばす
            file_path = f"{self.save_path}/{horse_id}.pickle"
            if os.path.isfile(file_path):
                continue
            peds_list = self.scrape(horse_id)
            self.save(peds_list, file_path)


# ============================================================================
# 4. 父・母父の過去成績
# ============================================================================


class PedigreeResults(BaseScraper):
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
        super().__init__()
        with open(horse_path, "rb") as f:
            self.horse_ids = pickle.load(f)
        self.save_path = save_path

    def df_process_sire_results(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df.columns = ["_".join(col).strip() for col in df.columns.values]
            df = df[df["年度_年度"] != "累計"]
        except Exception:
            df = None
        return df

    def scraping_result(self, horse_id: str) -> pd.DataFrame | None:
        """親馬の過去成績を取得する

        Args:
            horse_id (str): 親馬のID

        Raises:
            e: 親馬の過去成績の取得ができなかった場合はエラーをそのまま返す

        Returns:
            pd.DataFrame | None: 親馬の過去成績
        """
        isNotExist = True
        url = f"https://db.netkeiba.com/horse/{horse_id}"
        response = requests.get(url, headers=self.headers)
        response.encoding = "EUC-JP"
        html_string = io.StringIO(response.text)
        for i in range(1, 5):
            try:
                df = pd.read_html(html_string)[i]
                if df.columns[0] != "日付":
                    continue
                else:
                    isNotExist = False
                    break
            except IndexError:
                df = pd.DataFrame(0, index=range(1), columns=self.columns)
            except Exception as e:
                print(horse_id)
                raise e
        if isNotExist:
            df = pd.DataFrame(0, index=range(1), columns=self.columns)
        return df

    def scraping_sire_results(self, horse_id: str) -> pd.DataFrame | None:
        try:
            url = f"https://db.netkeiba.com/?pid=horse_sire&id={horse_id}&course=1&mode=1&type=0"
            response = requests.get(url, headers=self.headers)

            response.encoding = "EUC-JP"
            soup = bs(response.text, "html.parser")
            target_element1 = soup.select_one('table[summary="産駒成績"]')
            target_element2 = soup.select_one('table[summary="成績"]')
            df_tmp1 = (
                pd.read_html(io.StringIO(str(target_element1)))[0]
                if target_element1 is not None
                else None
            )
            df_tmp2 = (
                pd.read_html(io.StringIO(str(target_element2)))[0]
                if target_element2 is not None
                else None
            )
            df = {
                "sire_results": self.df_process_sire_results(df_tmp1),
                "sire_results_BMS": self.df_process_sire_results(df_tmp2),
            }
        except Exception as e:
            print(horse_id)
            raise e
        return df

    def scraping_sire_course(self, horse_id: str) -> pd.DataFrame | None:
        try:
            url = f"https://db.netkeiba.com/?pid=horse_sire&id={horse_id}&course=1&mode=1&type=1"
            response = requests.get(url, headers=self.headers)
            response.encoding = "EUC-JP"
            html_string = io.StringIO(response.text)
            try:
                tables = pd.read_html(html_string)
            except Exception:
                return [None, None]
            dfs = []
            first_loop = True
            for i in tables:
                tmp = i.copy()
                if tmp.iloc[0, 0] in ["輸入年", "供用開始年"]:
                    continue
                # マルチインデックスを1つにする
                tmp.columns = ["_".join(col).strip() for col in tmp.columns.values]
                # 累計の行は削除
                tmp = tmp[tmp["年度_年度"] != "累計"]
                # 初回ループはデータを格納する
                if first_loop:
                    first_loop = False
                    dfs.append(tmp)
                    continue
                # 次のブロックに移る
                elif tmp.columns[1] == "芝・良_1着":
                    dfs.append(tmp)
                    continue
                else:
                    dfs[-1] = pd.merge(dfs[-1], tmp, on="年度_年度", how="inner")
            # BeautifulSoupでパース
            soup = bs(response.text, "html.parser")
            # テーブルを含む要素がなければNoneを入れる
            target_element1 = soup.select_one('table[summary="産駒成績"]')
            target_element2 = soup.select_one('table[summary="成績"]')
            if target_element1 == None:
                dfs.insert(0, None)
            if target_element2 == None:
                dfs.append(None)
        except Exception as e:
            print(horse_id)
            raise e
        return dfs

    def scraping_sire_distance(self, horse_id: str) -> pd.DataFrame | None:
        try:
            url = f"https://db.netkeiba.com/?pid=horse_sire&id={horse_id}&course=1&mode=1&type=2"
            response = requests.get(url, headers=self.headers)
            response.encoding = "EUC-JP"
            html_string = io.StringIO(response.text)
            try:
                tables = pd.read_html(html_string)
            except Exception:
                return [None, None]
            dfs = []
            first_loop = True
            for i in tables:
                tmp = i.copy()
                if tmp.iloc[0, 0] in ["輸入年", "供用開始年"]:
                    continue
                # マルチインデックスを1つにする
                tmp.columns = ["_".join(col).strip() for col in tmp.columns.values]
                # 累計の行は削除
                tmp = tmp[tmp["年度_年度"] != "累計"]
                # 全データ型をintに変換
                tmp = tmp.astype(int)
                # 初回ループはデータを格納する
                if first_loop:
                    first_loop = False
                    dfs.append(tmp)
                    continue
                # 次のブロックに移る
                elif tmp.columns[1] == "-1400(芝)_1着":
                    dfs.append(tmp)
                    continue
                else:
                    dfs[-1] = pd.merge(dfs[-1], tmp, on="年度_年度", how="inner")
            # BeautifulSoupでパース
            soup = bs(response.text, "html.parser")
            # テーブルを含む要素がなければNoneを入れる
            target_element1 = soup.select_one('table[summary="産駒成績"]')
            target_element2 = soup.select_one('table[summary="成績"]')
            if target_element1 == None:
                dfs.insert(0, None)
            if target_element2 == None:
                dfs.append(None)
        except Exception as e:
            print(horse_id)
            raise e
        return dfs

    @classmethod
    def save(cls, data: dict | None, path: str, **kwargs) -> None:
        """レース結果を保存する

        Args:
            data (dict | None): レース結果の辞書
            path (str): 保存するファイルパス（拡張子を含む）
            **kwargs: その他のオプション引数
        """
        if data is None:
            return
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def scrape_save(self, sleep_rate=2) -> None:
        for horse_id in self.horse_ids:
            # 既にファイルが作成されていればスルー
            if os.path.isfile(f"{self.save_path}/{horse_id}.pickle"):
                continue
            start = time.perf_counter()
            df_result = self.scraping_result(horse_id)
            end = time.perf_counter()
            time.sleep(sleep_rate - (end - start))

            start = time.perf_counter()
            df_sire_results = self.scraping_sire_results(horse_id)
            end = time.perf_counter()
            time.sleep(sleep_rate - (end - start))

            start = time.perf_counter()
            df_sire_course = self.scraping_sire_course(horse_id)
            end = time.perf_counter()
            time.sleep(sleep_rate - (end - start))

            start = time.perf_counter()
            df_sire_distance = self.scraping_sire_distance(horse_id)
            end = time.perf_counter()
            time.sleep(sleep_rate - (end - start))

            df = {
                "result": df_result,
                "sire_results": df_sire_results,
                "sire_course": df_sire_course,
                "sire_distance": df_sire_distance,
            }
            file_path = f"{self.save_path}/{horse_id}.pickle"
            self.save(df, file_path)


# ============================================================================
# 5. 騎手情報
# ============================================================================


class JockeyResults(BaseScraper):
    def __init__(self, read_path: str) -> None:
        super().__init__()
        self.read_path = read_path
        self.jockey_ids = self.get_jockey_ids(self.read_path)

    def get_jockey_ids(self, path: str) -> list[str]:
        df = pd.read_pickle(path)
        return df["jockey_id"].to_list()

    def scrape_results(self, jockey_id: str) -> dict:
        url = f"https://db.netkeiba.com/jockey/result/{jockey_id}"
        response = requests.get(url, headers=self.headers)
        response.encoding = "EUC-JP"
        html_string = io.StringIO(response.text)
        soup = bs(response.text, "html.parser")
        name = soup.select_one("div.db_head_name.fc > h1").text.strip()
        clean_name = re.sub(r"\s*\(.*\)", "", name)
        try:
            tables = pd.read_html(html_string)[0]
        except Exception:
            tables = None
        if tables is not None:
            tables.columns = ["_".join(col).strip() for col in tables.columns.values]
        jockey = {"id": jockey_id, "name": clean_name, "results": tables}
        return jockey

    @classmethod
    def save(cls, data: dict, path: str, **kwargs) -> None:
        """騎手情報を保存する

        Args:
            data (dict): 騎手情報の辞書
            path (str): 保存するファイルパス（拡張子を含む）
            **kwargs: その他のオプション引数
        """
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def save_all(self, path: str) -> None:
        """すべての騎手情報を保存する

        Args:
            path (str): 保存先のディレクトリパス
        """
        for jockey_id in self.jockey_ids:
            file_name = f"{path}/{jockey_id}.pickle"
            if os.path.exists(file_name):
                continue
            result = self.scrape_results(jockey_id)
            self.save(result, file_name)
            time.sleep(1.5)
