import collections
import sys

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
    初期解を生成して親配列 parent を返す。
    方針:
      1. 頂点を A[v] の小さい順にソート
      2. まだどの木にも割り当てられていない頂点を根にして BFS。この時、高さが H を超えないように
    """
    # 1. 頂点を A[v] の小さい順にソート
    idx_sorted = sorted(range(N), key=lambda v: A[v])

    parent = [-1] * N
    assigned = [False] * N

    for v_root in idx_sorted:
        if not assigned[v_root]:
            parent[v_root] = -1
            assigned[v_root] = True

            # 2. まだどの木にも割り当てられていない頂点を根にして BFS。この時、高さが H を超えないように
            depth = {v_root: 0}
            queue = collections.deque([v_root])
            while queue:
                v = queue.popleft()
                if depth[v] < H:
                    for w in edges[v]:
                        if not assigned[w]:
                            assigned[w] = True
                            parent[w] = v
                            depth[w] = depth[v] + 1
                            queue.append(w)

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
    print(" ".join(map(str, parent)))

    return


if __name__ == '__main__':
    main()
