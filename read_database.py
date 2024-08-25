#!/usr/bin/env python3
import json
import pickle
import sys

DB_FILE = "database.pickle"


def search_algorithm_info(algorithm_name):
    with open(DB_FILE, "rb") as f:
        database = pickle.load(f)

    if algorithm_name not in database:
        return []
    return [{"problem_number": problem_number, "url": url} for problem_number, url in database[algorithm_name]]


def get_algorithm_list():
    with open(DB_FILE, "rb") as f:
        database = pickle.load(f)
    return list(database.keys())


if __name__ == "__main__":
    if len(sys.argv) == 1:
        results = get_algorithm_list()
        print(json.dumps(results, indent=2))
    elif len(sys.argv) == 2:
        algorithm_name = sys.argv[1]
        results = search_algorithm_info(algorithm_name)
        print(json.dumps(results, indent=2))
    else:
        print("アルゴリズム名を1つ入力")
        sys.exit(1)
