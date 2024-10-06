import heapq
import sys
from collections import defaultdict
from functools import lru_cache

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


# https://github.com/tatyam-prime/SortedSet/blob/main/SortedSet.py
import math
from bisect import bisect_left, bisect_right, insort
from typing import Generic, Iterable, Iterator, TypeVar, Union, List

T = TypeVar('T')


class SortedMultiset(Generic[T]):
    BUCKET_RATIO = 50
    REBUILD_RATIO = 170

    def _build(self, a=None) -> None:
        "Evenly divide `a` into buckets."
        if a is None: a = list(self)
        size = self.size = len(a)
        bucket_size = int(math.ceil(math.sqrt(size / self.BUCKET_RATIO)))
        self.a = [a[size * i // bucket_size: size * (i + 1) // bucket_size] for i in range(bucket_size)]

    def __init__(self, a: Iterable[T] = []) -> None:
        "Make a new SortedMultiset from iterable. / O(N) if sorted / O(N log N)"
        a = list(a)
        if not all(a[i] <= a[i + 1] for i in range(len(a) - 1)):
            a = sorted(a)
        self._build(a)

    def __iter__(self) -> Iterator[T]:
        for i in self.a:
            for j in i: yield j

    def __reversed__(self) -> Iterator[T]:
        for i in reversed(self.a):
            for j in reversed(i): yield j

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return "SortedMultiset" + str(self.a)

    def __str__(self) -> str:
        s = str(list(self))
        return "{" + s[1: len(s) - 1] + "}"

    def _find_bucket(self, x: T) -> List[T]:
        "Find the bucket which should contain x. self must not be empty."
        for a in self.a:
            if x <= a[-1]: return a
        return a

    def __contains__(self, x: T) -> bool:
        if self.size == 0: return False
        a = self._find_bucket(x)
        i = bisect_left(a, x)
        return i != len(a) and a[i] == x

    def count(self, x: T) -> int:
        "Count the number of x."
        return self.index_right(x) - self.index(x)

    def add(self, x: T) -> None:
        "Add an element. / O(√N)"
        if self.size == 0:
            self.a = [[x]]
            self.size = 1
            return
        a = self._find_bucket(x)
        insort(a, x)
        self.size += 1
        if len(a) > len(self.a) * self.REBUILD_RATIO:
            self._build()

    def discard(self, x: T) -> bool:
        "Remove an element and return True if removed. / O(√N)"
        if self.size == 0: return False
        a = self._find_bucket(x)
        i = bisect_left(a, x)
        if i == len(a) or a[i] != x: return False
        a.pop(i)
        self.size -= 1
        if len(a) == 0: self._build()
        return True

    def lt(self, x: T) -> Union[T, None]:
        "Find the largest element < x, or None if it doesn't exist."
        for a in reversed(self.a):
            if a[0] < x:
                return a[bisect_left(a, x) - 1]

    def le(self, x: T) -> Union[T, None]:
        "Find the largest element <= x, or None if it doesn't exist."
        for a in reversed(self.a):
            if a[0] <= x:
                return a[bisect_right(a, x) - 1]

    def gt(self, x: T) -> Union[T, None]:
        "Find the smallest element > x, or None if it doesn't exist."
        for a in self.a:
            if a[-1] > x:
                return a[bisect_right(a, x)]

    def ge(self, x: T) -> Union[T, None]:
        "Find the smallest element >= x, or None if it doesn't exist."
        for a in self.a:
            if a[-1] >= x:
                return a[bisect_left(a, x)]

    def __getitem__(self, x: int) -> T:
        "Return the x-th element, or IndexError if it doesn't exist."
        if x < 0: x += self.size
        if x < 0: raise IndexError
        for a in self.a:
            if x < len(a): return a[x]
            x -= len(a)
        raise IndexError

    def index(self, x: T) -> int:
        "Count the number of elements < x."
        ans = 0
        for a in self.a:
            if a[-1] >= x:
                return ans + bisect_left(a, x)
            ans += len(a)
        return ans

    def index_right(self, x: T) -> int:
        "Count the number of elements <= x."
        ans = 0
        for a in self.a:
            if a[-1] > x:
                return ans + bisect_right(a, x)
            ans += len(a)
        return ans


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


def my_bisect(lst, target):
    left = 0
    right = len(lst) - 1
    while left <= right:
        mid = (left + right) // 2
        if target <= lst[mid]:
            left = mid + 1
        else:
            right = mid - 1

    return left


# ============================================================================
def main():
    # WA
    n, m, k = INN()
    a = INN()

    if n == m:
        print(" ".join(["0" for _ in range(n)]))
        return

    copy_a = a.copy()
    copy_a.sort(reverse=True)
    cum_copy_a = [0]
    for i in range(n):
        cum_copy_a.append(cum_copy_a[i] + copy_a[i])
    sum_a = sum(a)

    def calc(i):
        left_ng = 0
        right_ok = k - sum_a

        while left_ng <= right_ok:
            mid = (left_ng + right_ok) // 2
            new_ai = a[i] + mid
            pos = my_bisect(copy_a, new_ai)
            if pos >= m:
                left_ng = mid + 1
                continue
            target = cum_copy_a[m] - cum_copy_a[pos]
            pre_pos = my_bisect(copy_a, a[i]) - 1
            needed = (new_ai + 1) * (m - pos)
            if pre_pos < m:
                target -= a[i]
                target += copy_a[m]
            if target + (k - sum_a - mid) >= needed:
                left_ng = mid + 1
            else:
                right_ok = mid - 1

        if left_ng > k - sum_a:
            return -1
        return left_ng

    ans = []
    for i in range(n):
        ans.append(calc(i))
    ans = map(str, ans)
    print(" ".join(ans))
    return


# ============================================================================

if __name__ == '__main__':
    main()
