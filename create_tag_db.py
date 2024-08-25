#!/usr/bin/env python3
import os
import pickle

TAG_FILENAME = "tags.txt"
TARGET_DIR = "./src"
DB_FILE = "database.pickle"


def create_database(folder_path):
    """初期の問題に対して、URLが正しくないことに注意"""
    database = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file == TAG_FILENAME:
                file_path = os.path.join(root, file)
                parts = root.split(os.path.sep)
                contest_name = parts[-2]
                problem_number = f"{contest_name}_{parts[-1]}"
                url = f"https://atcoder.jp/contests/{contest_name}/tasks/{problem_number}"
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        algorithm_name = line.strip()
                        if algorithm_name not in database:
                            database[algorithm_name] = []
                        database[algorithm_name].append((problem_number, url))
    return database


if __name__ == '__main__':
    database = create_database(TARGET_DIR)
    with open(DB_FILE, "wb") as f:
        pickle.dump(database, f)
