#!/usr/bin/env python3
import datetime
import os
import re
import shutil
import sys
from typing import List

CONTEST_PATH = "./src/contest/"
ARCHIVE_ROOT = "./src/"

def move_problem_files(contest: str) -> None:
    files = os.listdir(CONTEST_PATH)
    target = re.compile(r"\.py")

    contest_dir = os.path.join(ARCHIVE_ROOT, contest)
    os.makedirs(contest_dir, exist_ok=True)

    for f in files:
        if target.search(f):
            problem = f[:target.search(f).span()[0]]
            problem_dir = os.path.join(contest_dir, problem)
            os.makedirs(problem_dir, exist_ok=True)

            frm = os.path.join(CONTEST_PATH, f)
            to_filename = str(datetime.date.today().strftime("%Y%m%d") + ".py")
            to = os.path.join(problem_dir, to_filename)
            shutil.move(frm, to)


def main(args: List[str]) -> None:
    if len(args) != 2:
        raise Exception("コンテスト名を入力")

    contest = args[1]
    move_problem_files(contest)


if __name__ == '__main__':
    main(sys.argv)
