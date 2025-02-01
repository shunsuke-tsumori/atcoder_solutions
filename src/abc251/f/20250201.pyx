# cython: boundscheck=False, wraparound=False, cdivision=True
import sys
import collections
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def INN():
    return list(map(int, sys.stdin.readline().split()))

def main():
    cdef int n, m, i, u, v, nn, cn
    n, m = INN()
    paths = [[] for i in range(n)]
    for i in range(m):
        u, v = INN()
        u -= 1
        v -= 1
        paths[u].append(v)
        paths[v].append(u)

    t1 = []
    t1_visited = [False] * n

    def dfs(int crr):
        t1_visited[crr] = True
        for nn in paths[crr]:
            if not t1_visited[nn]:
                t1.append((crr + 1, nn + 1))
                dfs(nn)

    dfs(0)

    t2 = []
    t2_visited = [False] * n
    que = collections.deque()
    que.append(0)
    t2_visited[0] = True
    while que:
        cn = que.popleft()
        for nn in paths[cn]:
            if not t2_visited[nn]:
                t2_visited[nn] = True
                que.append(nn)
                t2.append((cn + 1, nn + 1))

    for i in range(n - 1):
        print(" ".join(map(str, t1[i])))
    for i in range(n - 1):
        print(" ".join(map(str, t2[i])))

if __name__ == '__main__':
    main()
