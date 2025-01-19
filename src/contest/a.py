import math
import random
import sys
import time

sys.setrecursionlimit(1000000)


def INN():
    return list(map(int, sys.stdin.readline().split()))


def IN_2(n: int) -> tuple[list[int], list[int]]:
    a, b = [], []
    for _ in range(n):
        ai, bi = INN()
        a.append(ai)
        b.append(bi)
    return a, b


# ============================================================================
def initial_solution(N, M, H, A, edges) -> list[int]:
    """
    初期解を DFS を用いて生成して親配列 parent を返す。
    方針:
      1. 頂点を A[v] の小さい順にソート
      2. まだどの木にも割り当てられていない頂点を root にして DFS。
         このとき、高さが H を超えないようにする。
    """
    # A[v] 昇順でソート
    idx_sorted = sorted(range(N), key=lambda v: A[v])

    parent = [-1] * N
    assigned = [False] * N
    depth = [-1] * N

    def dfs(v, d):
        """v を根として深さ d から DFS し、子の深さを d+1 にする。"""
        for w in edges[v]:
            if not assigned[w] and d < H:
                assigned[w] = True
                parent[w] = v
                depth[w] = d + 1
                dfs(w, d + 1)

    # まだ割り当てられていない頂点を root にして DFS
    for v_root in idx_sorted:
        if not assigned[v_root]:
            parent[v_root] = -1
            assigned[v_root] = True
            depth[v_root] = 0
            dfs(v_root, 0)

    return parent


def compute_depth_all(N, parent):
    """
    parent 配列から各頂点の高さ h_v を計算して返す。
    BFS を根ごとに行う。
    """
    depth = [-1] * N

    from collections import deque
    for v in range(N):
        if parent[v] == -1:
            depth[v] = 0
            queue = deque([v])
            while queue:
                cur = queue.popleft()
                for w in range(N):
                    if parent[w] == cur:
                        depth[w] = depth[cur] + 1
                        queue.append(w)

    return depth


def evaluate_solution(N, A, parent):
    """
    現在の解に対するスコアを計算。
    """
    depth = compute_depth_all(N, parent)
    total_value = 0
    for v in range(N):
        total_value += (depth[v] + 1) * A[v]
    return total_value


SEED = 0


def annealing(N, M, H, A, edges, parent, time_limit):
    """
    焼きなまし
    """
    random.seed(SEED)

    start_time = time.time()
    best_parent = parent[:]
    best_score = evaluate_solution(N, A, best_parent)

    current_parent = parent[:]
    current_score = best_score

    T0 = 1000.0
    T1 = 1e-2
    ITER_LIMIT = 10000

    for iteration in range(ITER_LIMIT):
        now = time.time()
        elapsed = now - start_time
        if elapsed > time_limit:
            break

        progress = elapsed / time_limit
        T = T0 * (T1 / T0) ** progress

        v = random.randrange(N)
        old_parent = current_parent[v]

        candidates = edges[v] + [-1]
        new_p = random.choice(candidates)
        if new_p == old_parent:
            continue

        if new_p != -1:
            temp = new_p
            cycle_found = False
            while temp != -1:
                if temp == v:
                    cycle_found = True
                    break
                temp = current_parent[temp]
            if cycle_found:
                continue

        current_parent[v] = new_p

        depth = compute_depth_all(N, current_parent)
        if max(depth) > H:
            current_parent[v] = old_parent
            continue

        new_score = 0
        for x in range(N):
            new_score += (depth[x] + 1) * A[x]

        diff = new_score - current_score
        if diff >= 0:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_parent = current_parent[:]
        else:
            prob = math.exp(diff / T)
            if random.random() < prob:
                current_score = new_score
            else:
                current_parent[v] = old_parent

    return best_parent


def main():
    N, M, H = INN()
    A = INN()
    edges = [[] for _ in range(N)]
    for _ in range(M):
        u, v = INN()
        edges[u].append(v)
        edges[v].append(u)
    x, y = IN_2(N)

    parent = initial_solution(N, M, H, A, edges)
    parent = annealing(N, M, H, A, edges, parent, time_limit=1.8)
    print(" ".join(map(str, parent)))
    print(evaluate_solution(N, A, parent))
    return


if __name__ == '__main__':
    main()
