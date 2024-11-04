import heapq
import sys
from collections import defaultdict
from functools import lru_cache
from sortedcontainers import SortedList, SortedSet, SortedDict

sys.setrecursionlimit(1000000)


def printYesNo(b: bool):
    print("Yes") if b else print("No")


def has_bit(num, shift):
    return (num >> shift) & 1 == 1


def IN():
    return int(sys.stdin.readline())


def INN():
    return list(map(int, sys.stdin.readline().split()))


def IS():
    return sys.stdin.readline().rstrip()


def ISS():
    return sys.stdin.readline().rstrip().split()


INF = 2 ** 60


def IN_2(n: int) -> tuple[list[int], list[int]]:
    a, b = [], []
    for _ in range(n):
        ai, bi = INN()
        a.append(ai)
        b.append(bi)
    return a, b


def IN_3(n: int) -> tuple[list[int], list[int], list[int]]:
    a, b, c = [], [], []
    for _ in range(n):
        ai, bi, ci = INN()
        a.append(ai)
        b.append(bi)
        c.append(ci)
    return a, b, c


def IN_4(n: int) -> tuple[list[int], list[int], list[int], list[int]]:
    a, b, c, d = [], [], [], []
    for _ in range(n):
        ai, bi, ci, di = INN()
        a.append(ai)
        b.append(bi)
        c.append(ci)
        d.append(di)
    return a, b, c, d


def bisect(a, n, x):
    left = 0
    right = n - 1
    while left <= right:
        mid = (left + right) // 2
        if a[mid] == x:
            return mid + 1
        if a[mid] < x:
            left = mid + 1
        else:
            right = mid - 1
    return -1


def divisors(n):
    l1, l2 = [], []
    i = 1
    while i * i <= n:
        if n % i == 0:
            l1.append(i)
            if i != n // i:
                l2.append(n // i)
        i += 1
    return l1 + l2[::-1]


def sieve(upper):
    is_prime = [True] * (upper + 1)
    is_prime[0] = False
    is_prime[1] = False
    for j in range(4, upper + 1, 2):
        is_prime[j] = False
    i = 3
    while i * i <= upper:
        if is_prime[i]:
            j = 2
            while i * j <= upper:
                is_prime[i * j] = False
                j += 1

        i += 1
    return is_prime


def primes(upper):
    # upperを含む
    ps = []
    if upper < 2:
        return ps
    is_prime = [True] * (int(upper) + 1)
    is_prime[0] = False
    is_prime[1] = False
    ps.append(2)
    j = 2
    while 2 * j <= upper:
        is_prime[2 * j] = False
        j += 1
    for i in range(3, int(upper) + 1, 2):
        if is_prime[i]:
            ps.append(i)
            j = 2
            while i * j <= upper:
                is_prime[i * j] = False
                j += 1
    return ps


def gcd(a, b):
    def _gcd(x, y):
        if x % y == 0:
            return y
        return _gcd(y, x % y)

    if a < b:
        a, b = b, a

    return _gcd(a, b)


def lcm(a, b):
    return a * b // gcd(a, b)


class UnionFind:
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n
        self.roots = set()
        for i in range(n):
            self.roots.add(i)

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x
        self.roots.discard(y)

    def size(self, x):
        return -self.parents[self.find(x)]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def group_count(self):
        return len(self.roots)

    def all_group_members(self):
        group_members = defaultdict(list)
        for member in range(self.n):
            group_members[self.find(member)].append(member)
        return group_members


class SegTree:
    def __init__(self, init_val, segfunc, ide_ele):
        n = len(init_val)
        self.segfunc = segfunc
        self.ide_ele = ide_ele
        self.num = 1 << (n - 1).bit_length()
        self.tree = [ide_ele] * 2 * self.num
        # 配列の値を葉にセット
        for i in range(n):
            self.tree[self.num + i] = init_val[i]
        # 構築していく
        for i in range(self.num - 1, 0, -1):
            self.tree[i] = self.segfunc(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, k, x):
        k += self.num
        self.tree[k] = x
        while k > 1:
            self.tree[k >> 1] = self.segfunc(self.tree[k], self.tree[k ^ 1])
            k >>= 1

    def query(self, l, r):
        res = self.ide_ele

        l += self.num
        r += self.num
        while l < r:
            if l & 1:
                res = self.segfunc(res, self.tree[l])
                l += 1
            if r & 1:
                res = self.segfunc(res, self.tree[r - 1])
            l >>= 1
            r >>= 1
        return res


class LazySegmentTree:
    def __init__(self, values, segment_function, ide_ele):
        n = len(values)
        self.segment_function = segment_function
        self.ide_ele = ide_ele
        self.num = 1 << (n - 1).bit_length()
        self.data = [ide_ele] * 2 * self.num
        self.lazy = [None] * 2 * self.num
        for i in range(n):
            self.data[self.num + i] = values[i]
        for i in range(self.num - 1, 0, -1):
            self.data[i] = self.segment_function(self.data[2 * i], self.data[2 * i + 1])

    def gindex(self, l, r):
        l += self.num
        r += self.num
        lm = l >> (l & -l).bit_length()
        rm = r >> (r & -r).bit_length()
        while l < r:
            if l <= lm:
                yield l
            if r <= rm:
                yield r
            r >>= 1
            l >>= 1
        while l:
            yield l
            l >>= 1

    def propagates(self, *ids):
        for i in reversed(ids):
            v = self.lazy[i]
            if v is None:
                continue
            self.lazy[2 * i] = v
            self.lazy[2 * i + 1] = v
            self.data[2 * i] = v
            self.data[2 * i + 1] = v
            self.lazy[i] = None

    def update(self, l, r, x):
        *ids, = self.gindex(l, r)
        self.propagates(*ids)
        l += self.num
        r += self.num
        while l < r:
            if l & 1:
                self.lazy[l] = x
                self.data[l] = x
                l += 1
            if r & 1:
                self.lazy[r - 1] = x
                self.data[r - 1] = x
            r >>= 1
            l >>= 1
        for i in ids:
            self.data[i] = self.segment_function(self.data[2 * i], self.data[2 * i + 1])

    def query(self, l, r):
        *ids, = self.gindex(l, r)
        self.propagates(*ids)
        res = self.ide_ele
        l += self.num
        r += self.num
        while l < r:
            if l & 1:
                res = self.segment_function(res, self.data[l])
                l += 1
            if r & 1:
                res = self.segment_function(res, self.data[r - 1])
            l >>= 1
            r >>= 1
        return res


mod = 998244353


def ext_gcd(a, b):
    if b == 0:
        return a, 1, 0
    d, y, x = ext_gcd(b, a % b)
    y -= (a // b) * x
    return d, x, y


def remainder(xm_list):
    x = 0
    d = 1
    for p, m in xm_list:
        g, a, b = ext_gcd(d, m)
        x, d = (m * b * x + d * a * p) // g, d * (m // g)
        x %= d
    return x, d


def combination(n, r, m):
    if n == r:
        return 1
    if r == 0:
        return 1
    a1 = 1
    for i in range(r):
        a1 *= (n - i)
        a1 %= m

    a2 = 1
    for i in range(1, r + 1):
        a2 *= i
        a2 %= m
    d = pow(a2, m - 2, m)
    return (a1 * d) % m


def compress(a):
    a_copy = a.copy()
    a_copy.sort()
    rank = {}
    i = 1
    rank[a_copy[0]] = 1
    for j in range(1, len(a)):
        if a_copy[j] != a_copy[j - 1]:
            i += 1
            rank[a_copy[j]] = i
    return [rank[a[i]] for i in range(len(a))]


class MaxFlowEdge:
    def __init__(self, to_node: int, capacity: int, rev_index: int):
        self.to_node = to_node
        self.capacity = capacity
        self.rev_index = rev_index


class MaxFlowSolver:
    """
    1-indexedとする
    """

    def __init__(self, n: int):
        """
        :param n ノード数
        """
        self.n = n
        self.graph: list[list[MaxFlowEdge]] = [[] for _ in range(n + 1)]
        self.visited = []

    def add_edge(self, from_node: int, to_node: int, capacity: int) -> None:
        graph_from_index = len(self.graph[from_node])
        graph_to_index = len(self.graph[to_node])
        self.graph[from_node].append(MaxFlowEdge(to_node, capacity, graph_to_index))
        self.graph[to_node].append(MaxFlowEdge(from_node, 0, graph_from_index))

    def __dfs(self, current: int, goal: int, flow: int) -> int:
        if current == goal:
            return flow
        self.visited[current] = True
        for edge in self.graph[current]:
            if not self.visited[edge.to_node] and edge.capacity > 0:
                next_flow = self.__dfs(edge.to_node, goal, min(flow, edge.capacity))
                if next_flow > 0:
                    edge.capacity -= next_flow
                    self.graph[edge.to_node][edge.rev_index].capacity += next_flow
                    return next_flow
        return 0

    def max_flow(self, start: int, goal: int) -> int:
        total = 0
        while True:
            self.visited = [False] * (self.n + 1)
            result = self.__dfs(start, goal, 10 ** 15)
            if result == 0:
                break
            total += result
        return total


# 要検証
class BIT:
    """
    1-indexed
    """

    def __init__(self, n):
        """
        :param n ノード数
        """
        self.size = n
        self.tree = [0] * (n + 1)

    def add(self, index, value):
        while index <= self.size:
            self.tree[index] += value
            index += index & -index

    def sum(self, index):
        """
        indexまでの総和を返す
        """
        total = 0
        while index > 0:
            total += self.tree[index]
            index -= index & -index

        return total


def dijkstra(n: int, paths, start: int, goal=None):
    """
    0-indexed
    """
    dists1 = [-1] * n
    visited = [False] * n
    que = [(0, start)]
    while len(que) > 0:
        cd, cn = heapq.heappop(que)
        if visited[cn]:
            continue
        if goal is not None and cn == goal:
            return cd
        visited[cn] = True
        dists1[cn] = cd
        for nn, nd in paths[cn]:
            if not visited[nn]:
                heapq.heappush(que, (nd + cd, nn))
    return dists1


def factorization(n):
    arr = []
    temp = n
    for i in range(2, int(-(-n ** 0.5 // 1)) + 1):
        if temp % i == 0:
            cnt = 0
            while temp % i == 0:
                cnt += 1
                temp //= i
            arr.append([i, cnt])

    if temp != 1:
        arr.append([temp, 1])

    if not arr:
        arr.append([n, 1])

    return arr


def rotate_matrix(matrix: list[list[any]], n: int) -> list[list[any]]:
    """
    2次元配列をn回90度時計回りに回転させた2次元配列を返す

    Args:
        matrix: 回転対象
        n: 回転数
    """
    n = n % 4
    rotated = matrix

    for _ in range(n):
        rotated = [list(row) for row in zip(*rotated)]
        rotated = [row[::-1] for row in rotated]

    return rotated

# ============================================================================
def main():

    n, q = INN()
    l = 0
    r = 1
    ans = 0

    def calc(tgt, other, dest):
        curr = 0
        is_ok = False
        i = tgt
        while True:
            if i == dest:
                is_ok = True
                break
            if i == other:
                break
            i += 1
            i %= n
            curr += 1
        if is_ok:
            return curr
        curr = 0
        i = tgt
        while True:
            if i == dest:
                break
            i -= 1
            i %= n
            curr += 1
        return curr

    for _ in range(q):
        h, t = ISS()
        t = int(t)
        t -= 1
        if h == "L":
            ans += calc(l, r, t)
            l = t
        else:
            ans += calc(r, l, t)
            r = t

    print(ans)

    return
# ============================================================================

if __name__ == '__main__':
    main()
