"""
スクレイピング実行用のスクリプト
"""

import os
import time
from tqdm import tqdm
from scraping import (
    RaceResult,
    HorseResult,
    PedigreeInfo,
    PedigreeResults,
    JockeyResults,
    get_all_files_in_directory,
    write_path,
    read_file,
)


# ============================================================================
# 1. レース結果のスクレイピング実行
# ============================================================================


def scrape_race_results():
    """レース結果をスクレイピングする"""
    # レースIDの構造：西暦(４桁)＋レース場ID＋開催回数＋何日目＋ラウンド
    race_id_list = []
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
                    race_id_list.append(race_id)

    year = os.listdir("../Raw-Data/Race-Results/")
    exits_file = get_all_files_in_directory("../Raw-Data/Race-Results/")
    skip_files_add = []
    skip_files = read_file("../tmp/skip_file_list.txt")

    for i in year:
        for j in tqdm(race_id_list, desc=f"{i}"):
            if f"../Raw-Data/Race-Results/{i}/{j}.pkl" in skip_files:
                continue
            elif f"../Raw-Data/Race-Results/{i}/{j}.pkl" in exits_file:
                skip_files_add.append(f"../Raw-Data/Race-Results/{i}/{j}.pkl")
                if len(skip_files_add) > 10:
                    write_path("../tmp/skip_file_list.txt", skip_files_add)
                    skip_files_add = []
                    skip_files = read_file("../tmp/skip_file_list.txt")
                continue
            try:
                time.sleep(1)
                scraping = RaceResult(f"https://db.netkeiba.com/race/{i}{j}/")
                scraping.save(
                    scraping.race_result,
                    f"../Raw-Data/Race-Results/{i}/{j}.pkl",
                )
                skip_files_add.append(f"../Raw-Data/Race-Results/{i}/{j}.pkl")
                if len(skip_files_add) > 10:
                    write_path("../tmp/skip_file_list.txt", skip_files_add)
                    skip_files_add = []
                    skip_files = read_file("../tmp/skip_file_list.txt")
            except IndexError:
                skip_files_add.append(f"../Raw-Data/Race-Results/{i}/{j}.pkl")
                if len(skip_files_add) > 10:
                    write_path("../tmp/skip_file_list.txt", skip_files_add)
                    skip_files_add = []
                    skip_files = read_file("../tmp/skip_file_list.txt")
                continue

            except Exception as e:
                print(f"{i}/{j}")
                raise e


def scrape_race_results_single_year(year: str):
    """特定の年のレース結果をスクレイピングする"""
    race_id_list = []
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
                    race_id_list.append(race_id)

    skip_race = []
    for i in tqdm(race_id_list):
        if os.path.isfile(f"../data/Race-Results/{year}/{i}.pkl"):
            continue
        try:
            scraping = RaceResult(f"https://db.netkeiba.com/race/{year}{i}/")
            scraping.save(
                scraping.race_result,
                f"../Raw-Data/Race-Results/{year}/{i}.pkl",
            )
            time.sleep(1)
        except IndexError:
            time.sleep(1)
            skip_race.append(f"{i}")
            continue
        except Exception as e:
            print(i)
            raise e
    with open("../skip-race.txt", "w") as f:
        for i in skip_race:
            f.write(f"{i}\n")


# ============================================================================
# 2. 競走馬の過去成績スクレイピング実行
# ============================================================================


def scrape_horse_results():
    """競走馬の過去成績をスクレイピングする"""
    file_list = get_all_files_in_directory("../Raw-Data/Race-Results/")
    save_path = "../Raw-Data/Horse-Results"
    for filename in tqdm(file_list):
        horse_results = HorseResult(f"{filename}", save_path=save_path)
        horse_results.scraping_and_save()


# ============================================================================
# 3. 血統情報スクレイピング実行
# ============================================================================


def scrape_pedigree_info():
    """血統情報をスクレイピングする"""
    file_list = get_all_files_in_directory("../Raw-Data/Race-Results/")
    did_peds_list = read_file("../tmp/did_peds_list.txt")
    save_path = "../Raw-Data/Pedigree"
    for filename in tqdm(file_list):
        # 既にファイルが存在しているなら飛ばす
        if filename in did_peds_list:
            continue
        else:
            try:
                pedigree = PedigreeInfo(f"{filename}", save_path=save_path)
                pedigree.scrape_save()
            except Exception as e:
                print(filename)
                raise e


# ============================================================================
# 4. 父・母父の過去成績スクレイピング実行
# ============================================================================


def scrape_pedigree_results():
    """父・母父の過去成績をスクレイピングする"""
    dir_path = "../Raw-Data/Pedigree"
    save_path = "../Raw-Data/Pedigree-Results"
    dir_list = os.listdir("../Raw-Data/Pedigree/")
    for filename in tqdm(dir_list):
        try:
            pedigree_results = PedigreeResults(
                f"{dir_path}/{filename}", save_path=save_path
            )
            pedigree_results.scrape_save(sleep_rate=3)
        except Exception as e:
            print(filename)
            raise e


# ============================================================================
# 5. 騎手情報スクレイピング実行
# ============================================================================


def scrape_jockey_results():
    """騎手情報をスクレイピングする"""
    # レース結果のpklファイルを読み込む
    race_results_dir = "../Raw-Data/Race-Results"
    race_results_files = []

    # サブディレクトリを含めて全てのpklファイルを取得
    for root, dirs, files in os.walk(race_results_dir):
        for file_name in files:
            if file_name.endswith(".pkl"):
                race_results_files.append(os.path.join(root, file_name))

    # 各ファイルを読み込んでDataFrameを取得
    for file_name in tqdm(race_results_files):
        try:
            jockey_results = JockeyResults(file_name)
            jockey_results.save_all("../Raw-Data/Jockey-Results/")
        except Exception as e:
            print(file_name)
            raise e


# ============================================================================
# メイン実行
# ============================================================================

if __name__ == "__main__":
    # 実行したいスクレイピング関数のコメントを外す
    # scrape_race_results()
    # scrape_race_results_single_year("2022")
    # scrape_horse_results()
    # scrape_pedigree_info()
    # scrape_pedigree_results()
    # scrape_jockey_results()
    pass
