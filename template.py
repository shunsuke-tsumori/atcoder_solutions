import heapq
import math
import sys
from collections import defaultdict
from functools import lru_cache
from sortedcontainers import SortedList, SortedSet, SortedDict

sys.setrecursionlimit(1000000)


def printYesNo(b: bool):
    print("Yes") if b else print("No")


def has_bit(num: int, shift: int) -> bool:
    """
    指定されたビット位置にビットが立っているかを判定します。

    この関数は、整数 `num` の `shift` ビット目が1であるかどうかを確認します。
    ビット位置は0から始まり、0が最下位ビットを表します。

    Args:
        num (int): 判定対象の整数。
        shift (int): チェックするビットの位置。0が最下位ビットを表します。

    Returns:
        bool: 指定されたビット位置にビットが立っている場合はTrue、そうでない場合はFalse。
    """
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


def divisors(n: int) -> list[int]:
    """
    指定された整数の約数を全て取得します。

    この関数は、与えられた整数 `n` の全ての正の約数をリストとして返します。
    約数は昇順に並べられます。

    Args:
        n (int): 約数を求めたい正の整数。

    Returns:
        list[int]: 整数 `n` の全ての正の約数を含むリスト。
    """
    l1, l2 = [], []
    i = 1
    while i * i <= n:
        if n % i == 0:
            l1.append(i)
            if i != n // i:
                l2.append(n // i)
        i += 1
    return l1 + l2[::-1]


def sieve(upper: int) -> list[bool]:
    """
    エラトステネスの篩を用いて、指定された範囲までの素数を判定します。

    この関数は、与えられた整数 `upper` までの各数が素数かどうかを示すブール値のリストを返します。
    リストのインデックス `i` が素数であれば `is_prime[i]` は `True`、そうでなければ `False` になります。

    Args:
        upper (int): 素数を判定する上限値。0から `upper` までの整数について判定します。

    Returns:
        list[bool]: 各インデックス `i` に対して、`i` が素数であれば `True`、そうでなければ `False`。
    """
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


def primes(upper: int) -> list[int]:
    """
    エラトステネスの篩を用いて、指定された範囲までの素数を取得します。

    この関数は、与えられた整数 `upper` 以下の全ての素数をリストとして返します。
    範囲は0から `upper` までを含みます。

    Args:
        upper (int): 素数を取得する上限値。0以上の整数を指定します。

    Returns:
        list[int]: 0から `upper` までの全ての素数を含むリスト。
    """
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


class UnionFind:
    def __init__(self, n: int):
        """
        指定された数の要素でUnion-Find構造を初期化します。各要素は初めは個別の集合に属します。

        Args:
            n (int): 要素の数。要素は0からn-1までの整数で表されます。
        """
        self.n = n
        self.parents = [-1] * n
        self.roots = set()
        for i in range(n):
            self.roots.add(i)

    def find(self, x: int) -> int:
        """
        要素xのルートを見つけます。経路圧縮を行います。

        Args:
            x (int): ルートを見つけたい要素のインデックス。

        Returns:
            int: 要素xが属する集合のルートのインデックス。
        """
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x: int, y: int):
        """
        要素xと要素yが属する集合を統合します。

        Args:
            x (int): 統合したい要素のインデックス。
            y (int): 統合したい要素のインデックス。
        """
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x
        self.roots.discard(y)

    def size(self, x: int) -> int:
        """
        要素xが属する集合のサイズを返します。

        Args:
            x (int): 集合のサイズを知りたい要素のインデックス。

        Returns:
            int: 要素xが属する集合のサイズ。
        """
        return -self.parents[self.find(x)]

    def same(self, x: int, y: int) -> bool:
        """
        要素xと要素yが同じ集合に属するかどうかを判定します。

        Args:
            x (int): 判定したい要素のインデックス。
            y (int): 判定したい要素のインデックス。

        Returns:
            bool: 要素xと要素yが同じ集合に属する場合はTrue、そうでない場合はFalse。
        """
        return self.find(x) == self.find(y)

    def members(self, x: int) -> list[int]:
        """
        要素xが属する集合の全メンバーをリストで返します。

        Args:
            x (int): 集合のメンバーを取得したい要素のインデックス。

        Returns:
            list[int]: 要素xが属する集合の全メンバーのインデックスのリスト。
        """
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def group_count(self) -> int:
        """
        現在の集合の数を返します。

        Returns:
            int: 現在の集合の数。
        """
        return len(self.roots)

    def all_group_members(self) -> dict[int, list[int]]:
        """
        全ての集合のメンバーを辞書で返します。

        Returns:
            dict[int, list[int]]: 各キーが集合のルート、値がその集合のメンバーのリストとなる辞書。
        """
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


class BIT:
    """
    1-indexed の Binary Indexed Tree（Fenwick Tree）。

    このクラスは、数列の要素の更新と区間の累積和を効率的に計算するためのデータ構造です。
    インデックスは1から始まります。
    """

    def __init__(self, n: int) -> None:
        """
        コンストラクタ。

        指定されたサイズ `n` のBITを初期化します。初期状態では全ての要素は0に設定されます。

        Args:
            n (int): ノード数。BITは1から `n` までのインデックスを扱います。
        """
        self.size: int = n
        self.tree: list[int] = [0] * (n + 1)

    def add(self, index: int, value: int) -> None:
        """
        指定したインデックスに値を加算します。

        このメソッドは、インデックス `index` に `value` を加算し、関連する累積和を更新します。

        Args:
            index (int): 値を加算する対象のインデックス（1から始まる）。
            value (int): 加算する値。
        """
        while index <= self.size:
            self.tree[index] += value
            index += index & -index

    def sum(self, index: int) -> int:
        """
        指定したインデックスまでの累積和を返します。

        このメソッドは、インデックス `index` までの要素の総和を計算して返します。

        Args:
            index (int): 累積和を計算する対象のインデックス（1から始まる）。

        Returns:
            int: インデックス `index` までの総和。
        """
        total: int = 0
        while index > 0:
            total += self.tree[index]
            index -= index & -index
        return total


def count_inversions(array: list[int]) -> int:
    """
    配列の転倒数（Inversion Count）を計算します。

    転倒数とは、配列内で前にある要素が後ろの要素よりも大きいペアの数を指します。
    この関数では、Binary Indexed Tree（BIT）を使用して効率的に転倒数を計算します。

    Args:
        array (List[int]): 転倒数を計算したい整数の配列。

    Returns:
        int: 配列の転倒数。
    """
    if len(array) == 0:
        return 0

    sorted_unique = sorted(list(set(array)))
    rank = {num: idx + 1 for idx, num in enumerate(sorted_unique)}

    bit = BIT(len(sorted_unique))

    inversion_count = 0
    for num in reversed(array):
        r = rank[num]
        inversion_count += bit.sum(r - 1)
        bit.add(r, 1)

    return inversion_count


def dijkstra(
        n: int,
        paths: list[list[tuple[int, int]]],
        start: int,
        goal: int | None = None
) -> int | list[int]:
    """
    ダイクストラ法を用いて、指定されたグラフ上の最短経路を計算します。

    この関数は、ノード数 `n` と各ノードからの接続情報 `paths` を基に、
    開始ノード `start` から他のノードへの最短距離を計算します。
    オプションで目標ノード `goal` を指定すると、そのノードへの最短距離のみを返します。

    Args:
        n (int): グラフのノード数。ノードは0からn-1までの整数で表されます。
        paths (list[list[tuple[int, int]]]): 各ノードから接続されているノードとその距離のリスト。
            例えば、paths[u] に (v, w) が含まれている場合、ノードuからノードvへの距離はwです。
        start (int): 最短経路の開始ノードのインデックス。
        goal (Optional[int], optional): 最短経路の終了ノードのインデックス。指定しない場合は
            全てのノードへの最短距離をリストで返します。デフォルトは `None`。

    Returns:
        Union[int, list[int]]:
            - `goal` が指定された場合は、開始ノードから `goal` ノードへの最短距離を返します。
            - `goal` が指定されていない場合は、開始ノードから全てのノードへの最短距離を
              各ノードのインデックスに対応するリストとして返します。
              到達不可能なノードについては -1 が設定されます。
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


def factorization(n: int) -> list[list[int]]:
    """
    指定された整数の素因数分解を行います。

    この関数は、与えられた整数 `n` を素因数分解し、各素因数とその指数をペアとしたリストを返します。
    結果のリストは、各要素が `[素因数, 指数]` の形式となっています。

    Args:
        n (int): 素因数分解を行いたい正の整数。

    Returns:
        list[list[int]]: 素因数とその指数のペアを含むリスト。
                         例えば、n=12 の場合、[[2, 2], [3, 1]] を返します。
    """
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


from typing import Any


def create_matrix(default_value: Any, rows: int, columns: int) -> list[list[Any]]:
    """
    指定されたサイズとデフォルト値で2次元の行列を作成します。

    この関数は、`rows` 行 `columns` 列の2次元リスト（行列）を作成し、
    各要素を `default_value` で初期化します。

    Args:
        default_value (Any): 行列の各要素に設定するデフォルト値。
        rows (int): 行列の行数。
        columns (int): 行列の列数。

    Returns:
        list[list[Any]]: 指定されたサイズとデフォルト値で初期化された2次元リスト。
    """
    return [[default_value] * columns for _ in range(rows)]


# ============================================================================
def main():
    return
# ============================================================================

if __name__ == '__main__':
    main()
