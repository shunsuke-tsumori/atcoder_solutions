#!/usr/bin/env python3
import os
import pathlib
import shutil
import sys
from typing import List

CONTEST_PATH = "./src/contest/"
TEMPLATE_PATH = "./template.py"


def create_problem_files(num_problems: int) -> None:
    os.makedirs(CONTEST_PATH, exist_ok=True)
    pathlib.Path(CONTEST_PATH + "test.txt").touch()

    for i in range(num_problems):
        problem_letter = chr(i + ord('a'))
        problem_file_path = os.path.join(CONTEST_PATH, f"{problem_letter}.py")
        if not os.path.exists(problem_file_path):
            shutil.copy(TEMPLATE_PATH, problem_file_path)
        else:
            print(f"{problem_file_path} exists")


def main(args: List[str]) -> None:
    if len(args) != 2:
        print("問題数を入力")
        sys.exit(1)

    num_problems = int(args[1])
    create_problem_files(num_problems)


if __name__ == '__main__':
    main(sys.argv)
