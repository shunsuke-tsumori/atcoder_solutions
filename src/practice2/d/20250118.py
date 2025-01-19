import heapq
import math
import sys
from collections import defaultdict
from functools import lru_cache

from sortedcontainers import SortedList, SortedSet, SortedDict

sys.setrecursionlimit(1000000)

#####################################################
# CONSTS
#####################################################
INF = 2 ** 60
MODULO = 998244353
LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
# ROLLING_HASH_MOD = 2305843009213693951
ROLLING_HASH_MOD = 8128812800000059


#####################################################
# I/O
#####################################################
def Takahashi():
    print("Takahashi")


def Aoki():
    print("Aoki")


def IN():
    return int(sys.stdin.readline())


def INN():
    return list(map(int, sys.stdin.readline().split()))


def IS():
    return sys.stdin.readline().rstrip()


def ISS():
    return sys.stdin.readline().rstrip().split()


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


#####################################################
# Bitwise Calculations
#####################################################
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


#####################################################
# Math
#####################################################
def floor_sum(n: int, m: int, a: int, b: int) -> int:
    """
    floor_sum(n, m, a, b) は、以下の総和を効率的に計算します:
        S = sum_{i=0}^{n-1} floor((a*i + b) / m)

    大きな n に対しても高速に計算可能。（O(log(a)+log(m))程度らしい。）

    パラメータ
    ----------
    n : int
        総和を取るときの上限（i の最大値は n-1）。
    m : int
        分母となる値。
    a : int
        i と掛け合わせる係数。
    b : int
        分母 m で割る前に加算される定数項。

    戻り値
    -------
    ans : int
        sum_{i=0}^{n-1} floor((a*i + b)/m) の計算結果。
    """

    ans = 0

    if a >= m:
        ans += (n - 1) * n * (a // m) // 2
        a %= m

    if b >= m:
        ans += n * (b // m)
        b %= m

    y_max = (a * n + b) // m
    x_max = y_max * m - b

    if y_max == 0:
        return ans

    ans += (n - (x_max + a - 1) // a) * y_max
    ans += floor_sum(y_max, a, m, (a - x_max % a) % a)

    return ans


#####################################################
# Number Theory
#####################################################
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


def ext_gcd(a: int, b: int) -> tuple[int, int, int]:
    """
    拡張ユークリッドの互除法を用いて、2つの整数 a と b の最大公約数と係数を求めます。

    この関数は、最大公約数 d と、d = a * x + b * y を満たす整数 x, y をタプル (d, x, y) として返します。

    パラメータ:
        a (int): 第1の整数
        b (int): 第2の整数

    戻り値:
        tuple[int, int, int]: 最大公約数 d、係数 x、係数 y のタプル

    例:
        >>> ext_gcd(30, 20)
        (10, 1, -1)
    """
    if b == 0:
        return a, 1, 0
    d, y, x = ext_gcd(b, a % b)
    y -= (a // b) * x
    return d, x, y


def remainder(xm_list: list[tuple[int, int]]) -> tuple[int, int]:
    """
    中国の剰余定理を用いて、一連の合同式を解きます。

    この関数は、与えられたリスト `xm_list` に含まれる (余り p, 法 m) のペアに基づいて、
    すべての合同式 x ≡ p mod m を満たす最小の非負整数 x と、その周期 d を返します。

    パラメータ:
        xm_list (list[tuple[int, int]]): 各要素が (余り p, 法 m) のタプルであるリスト

    戻り値:
        tuple[int, int]: 最小の非負整数 x とその周期 d のタプル

    例:
        >>> remainder([(2, 3), (3, 5), (2, 7)])
        (23, 105)
    """
    x = 0
    d = 1
    for p, m in xm_list:
        g, a, b = ext_gcd(d, m)
        x, d = (m * b * x + d * a * p) // g, d * (m // g)
        x %= d
    return x, d


def combination(n: int, r: int, m: int) -> int:
    """
    組み合わせ C(n, r) を法 m で計算します。

    この関数は、分子と分母から共通の因数を取り除くことで、
    逆元が存在しない場合でも正しい結果を計算します。

    パラメータ:
        n (int): 選ぶ元の総数
        r (int): 選ぶ元の数
        m (int): 法（正の整数）

    戻り値:
        int: 組み合わせ C(n, r) を m で割った余り
    """
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


#####################################################
# Union Find / Disjoint Set Union
#####################################################
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


#####################################################
# SegTree
#####################################################
from typing import Callable, TypeVar, Generator

T = TypeVar('T')
"""
Tはセグメントツリーが扱う要素の型を表します。セグメントツリーは任意のデータ型に対して汎用的に使用できます。
例えば、整数、浮動小数点数、文字列、カスタムオブジェクトなどが含まれます。
"""


class SegTree:
    """
    セグメントツリー（Segment Tree）データ構造。

    このクラスは、数列の特定の区間に対する演算（例えば、和、最小値、最大値など）を効率的に計算・更新するためのデータ構造です。
    0-indexed で動作し、初期化時に指定された演算と単位元に基づいてツリーを構築します。
    """

    def __init__(self, init_val: list[T], segfunc: Callable[[T, T], T], ide_ele: T):
        """
        コンストラクタ。

        指定された初期値リスト、セグメント関数、および単位元を用いてセグメントツリーを初期化します。

        Args:
            init_val (list[T]): セグメントツリーの初期値となるリスト。
            segfunc (Callable[[T, T], T]): セグメントツリーで使用する演算関数。例として、和を計算する場合は `lambda x, y: x + y`。
            ide_ele (T): セグメントツリーの単位元。例えば和の場合は `0`、最小値の場合は `float('inf')` など。
        """
        n = len(init_val)
        self.n = n
        self.segfunc = segfunc
        self.ide_ele = ide_ele
        self.num = 1 << (n - 1).bit_length()
        self.tree = [ide_ele] * (2 * self.num)
        for i in range(n):
            self.tree[self.num + i] = init_val[i]
        for i in range(self.num - 1, 0, -1):
            self.tree[i] = self.segfunc(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, k: int, x: T) -> None:
        """
        指定したインデックスの値を更新します。

        Args:
            k (int): 更新対象のインデックス（0-indexed）。
            x (T): 新しい値。
        """
        k += self.num
        self.tree[k] = x
        while k > 1:
            self.tree[k >> 1] = self.segfunc(self.tree[k], self.tree[k ^ 1])
            k >>= 1

    def query(self, l: int, r: int) -> T:
        """
        指定した区間 [l, r) に対する演算結果を取得します。

        Args:
            l (int): クエリの開始インデックス（0-indexed、含む）。
            r (int): クエリの終了インデックス（0-indexed、含まない）。

        Returns:
            T: 指定区間に対する演算結果。
        """
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

    def max_right(self, left: int, f: Callable[[T], bool]) -> int:
        """
        条件 f を満たす最大の right を探して返す。
        [left, right) の区間全体が f を満たす最大の right。
        つまり、left <= right <= self.n かつ
        すべての区間 [left, x) (left <= x <= right) で f(query(left, x)) が True となり、
        [left, right+1) では False になるような right を返す。

        Args:
            left (int): 探索を開始する左端(0-indexed)
            f (Callable[[T], bool]): 単調性を持つ判定関数。f(x)がTrueなら右へ伸ばす、Falseなら止まる。

        Returns:
            int: 条件を満たす最大の right (0 <= right <= self.n)
        """
        if left == self.n:
            return self.n

        left += self.num
        sm = self.ide_ele
        first = True
        while first or (left & -left) != left:
            first = False
            while left % 2 == 0:
                left >>= 1
            if not f(self.segfunc(sm, self.tree[left])):
                while left < self.num:
                    left <<= 1
                    if f(self.segfunc(sm, self.tree[left])):
                        sm = self.segfunc(sm, self.tree[left])
                        left += 1
                return left - self.num
            sm = self.segfunc(sm, self.tree[left])
            left += 1

        return self.n

    def min_left(self, right: int, f: Callable[[T], bool]) -> int:
        """
        条件 f を満たす最小の left を探して返す。
        [left, right) の区間全体が f を満たす最小の left。
        つまり、0 <= left <= right <= self.n かつ
        すべての区間 [x, right) (left <= x <= right) で f(query(x, right)) が True となり、
        [left-1, right) では False になるような left を返す。

        Args:
            right (int): 探索を開始する右端(0-indexed)
            f (Callable[[T], bool]): 単調性を持つ判定関数。

        Returns:
            int: 条件を満たす最小の left (0 <= left <= right)
        """
        if right == 0:
            return 0

        right += self.num
        sm = self.ide_ele
        first = True
        while first or (right & -right) != right:
            first = False
            right -= 1
            while right > 1 and right % 2 == 1:
                right >>= 1
            if not f(self.segfunc(self.tree[right], sm)):
                while right < self.num:
                    right = 2 * right + 1
                    if f(self.segfunc(self.tree[right], sm)):
                        sm = self.segfunc(self.tree[right], sm)
                        right -= 1
                return right + 1 - self.num
            sm = self.segfunc(self.tree[right], sm)
        return 0


class LazySegmentTree:
    """
    セグメントツリー（Segment Tree）に遅延伝搬（Lazy Propagation）を組み込んだデータ構造。

    このクラスは、数列の特定の区間に対する演算（例えば、和、最小値、最大値など）を効率的に計算・更新するためのデータ構造です。
    ジェネリック型 `T` を使用することで、任意のデータ型に対して汎用的に動作します。0-indexed で動作し、初期化時に指定された演算と単位元に基づいてツリーを構築します。
    """

    def __init__(self, values: list[T], segment_function: Callable[[T, T], T], ide_ele: T):
        """
        コンストラクタ。

        指定された初期値リスト、セグメント関数、および単位元を用いてセグメントツリーを初期化します。
        初期値リストの長さに基づいてツリーのサイズを決定し、セグメント関数を用いてツリーを構築します。

        Args:
            values (list[T]): セグメントツリーの初期値となるリスト。
            segment_function (Callable[[T, T], T]): セグメントツリーで使用する演算関数。例として、和を計算する場合は `lambda x, y: x + y`。
            ide_ele (T): セグメントツリーの単位元。例えば和の場合は `0`、最小値の場合は `float('inf')` など。
        """
        n = len(values)
        self.segment_function = segment_function
        self.ide_ele = ide_ele
        self.num = 1 << (n - 1).bit_length()
        self.data = [ide_ele] * (2 * self.num)
        self.lazy = [None] * (2 * self.num)
        for i in range(n):
            self.data[self.num + i] = values[i]
        for i in range(self.num - 1, 0, -1):
            self.data[i] = self.segment_function(self.data[2 * i], self.data[2 * i + 1])

    def gindex(self, l: int, r: int) -> Generator[int, None, None]:
        """
        更新またはクエリを行う際に必要なノードのインデックスを生成します。

        Args:
            l (int): クエリまたは更新の開始インデックス（0-indexed、含む）。
            r (int): クエリまたは更新の終了インデックス（0-indexed、含まない）。

        Yields:
            int: 処理対象となるノードのインデックス。
        """
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

    def propagates(self, *ids: int) -> None:
        """
        指定されたノードの遅延伝搬を行います。

        Args:
            *ids (int): 遅延伝搬を行うノードのインデックス。
        """
        for i in reversed(ids):
            v = self.lazy[i]
            if v is None:
                continue
            self.lazy[2 * i] = v
            self.lazy[2 * i + 1] = v
            self.data[2 * i] = v
            self.data[2 * i + 1] = v
            self.lazy[i] = None

    def update(self, l: int, r: int, x: T) -> None:
        """
        指定した区間 [l, r) に対して値 `x` を一括で更新します。

        Args:
            l (int): 更新対象の区間の開始インデックス（0-indexed、含む）。
            r (int): 更新対象の区間の終了インデックス（0-indexed、含まない）。
            x (T): 更新する値。
        """
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

    def query(self, l: int, r: int) -> T:
        """
        指定した区間 [l, r) に対する演算結果を取得します。

        Args:
            l (int): クエリの開始インデックス（0-indexed、含む）。
            r (int): クエリの終了インデックス（0-indexed、含まない）。

        Returns:
            T: 指定区間に対する演算結果。
        """
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


#####################################################
# Compress
#####################################################
def compress(a: list[int]) -> list[int]:
    """
    座標圧縮を行います。

    この関数は、リスト `a` 内の要素をソートし、それぞれの要素に対して
    一意のランクを割り当てます。元のリストの要素をそのランクに置き換えた
    新しいリストを返します。ランクは1から始まります。

    パラメータ:
        a (list[int]): 座標圧縮を行う整数のリスト。

    戻り値:
        list[int]: 元のリストの各要素が対応するランクに置き換えられたリスト。

    例:
        >>> compress([40, 10, 20, 20, 30])
        [4, 1, 2, 2, 3]
    """
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


#####################################################
# Max Flow
#####################################################
from typing import NamedTuple, Optional, List, cast


class MFGraph:
    class Edge(NamedTuple):
        src: int
        dst: int
        cap: int
        flow: int

    class _Edge:
        def __init__(self, dst: int, cap: int) -> None:
            self.dst = dst
            self.cap = cap
            self.rev: Optional[MFGraph._Edge] = None

    def __init__(self, n: int) -> None:
        self._n = n
        self._g: List[List[MFGraph._Edge]] = [[] for _ in range(n)]
        self._edges: List[MFGraph._Edge] = []

    def add_edge(self, src: int, dst: int, cap: int) -> int:
        assert 0 <= src < self._n
        assert 0 <= dst < self._n
        assert 0 <= cap
        m = len(self._edges)
        e = MFGraph._Edge(dst, cap)
        re = MFGraph._Edge(src, 0)
        e.rev = re
        re.rev = e
        self._g[src].append(e)
        self._g[dst].append(re)
        self._edges.append(e)
        return m

    def get_edge(self, i: int) -> Edge:
        assert 0 <= i < len(self._edges)
        e = self._edges[i]
        re = cast(MFGraph._Edge, e.rev)
        return MFGraph.Edge(
            re.dst,
            e.dst,
            e.cap + re.cap,
            re.cap
        )

    def edges(self) -> List[Edge]:
        return [self.get_edge(i) for i in range(len(self._edges))]

    def change_edge(self, i: int, new_cap: int, new_flow: int) -> None:
        assert 0 <= i < len(self._edges)
        assert 0 <= new_flow <= new_cap
        e = self._edges[i]
        e.cap = new_cap - new_flow
        assert e.rev is not None
        e.rev.cap = new_flow

    def flow(self, s: int, t: int, flow_limit: Optional[int] = None) -> int:
        assert 0 <= s < self._n
        assert 0 <= t < self._n
        assert s != t
        if flow_limit is None:
            flow_limit = cast(int, sum(e.cap for e in self._g[s]))

        current_edge = [0] * self._n
        level = [0] * self._n

        def fill(arr: List[int], value: int) -> None:
            for i in range(len(arr)):
                arr[i] = value

        def bfs() -> bool:
            fill(level, self._n)
            queue = []
            q_front = 0
            queue.append(s)
            level[s] = 0
            while q_front < len(queue):
                v = queue[q_front]
                q_front += 1
                next_level = level[v] + 1
                for e in self._g[v]:
                    if e.cap == 0 or level[e.dst] <= next_level:
                        continue
                    level[e.dst] = next_level
                    if e.dst == t:
                        return True
                    queue.append(e.dst)
            return False

        def dfs(lim: int) -> int:
            stack = []
            edge_stack: List[MFGraph._Edge] = []
            stack.append(t)
            while stack:
                v = stack[-1]
                if v == s:
                    flow = min(lim, min(e.cap for e in edge_stack))
                    for e in edge_stack:
                        e.cap -= flow
                        assert e.rev is not None
                        e.rev.cap += flow
                    return flow
                next_level = level[v] - 1
                while current_edge[v] < len(self._g[v]):
                    e = self._g[v][current_edge[v]]
                    re = cast(MFGraph._Edge, e.rev)
                    if level[e.dst] != next_level or re.cap == 0:
                        current_edge[v] += 1
                        continue
                    stack.append(e.dst)
                    edge_stack.append(re)
                    break
                else:
                    stack.pop()
                    if edge_stack:
                        edge_stack.pop()
                    level[v] = self._n
            return 0

        flow = 0
        while flow < flow_limit:
            if not bfs():
                break
            fill(current_edge, 0)
            while flow < flow_limit:
                f = dfs(flow_limit - flow)
                flow += f
                if f == 0:
                    break
        return flow

    def min_cut(self, s: int) -> List[bool]:
        visited = [False] * self._n
        stack = [s]
        visited[s] = True
        while stack:
            v = stack.pop()
            for e in self._g[v]:
                if e.cap > 0 and not visited[e.dst]:
                    visited[e.dst] = True
                    stack.append(e.dst)
        return visited


#####################################################
# Graph
#####################################################
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


#####################################################
# Matrix
#####################################################
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


DIR4 = [
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1)
]
"""上右下左"""
DIR8 = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1)
]
"""上から時計回り"""


#####################################################
# Run Length Encoding
#####################################################
def run_length_encoding(s: str) -> list[(str, int)]:
    """
    与えられた文字列を連長圧縮します。

    引数:
        s (str): エンコード対象の文字列。

    戻り値:
        list[(str, int)]: 各文字とその連続出現回数を持つタプルのリスト。

    使用例:
        >>> run_length_encoding("AAAABBBCCDAA")
        [('A', 4), ('B', 3), ('C', 2), ('D', 1), ('A', 2)]
    """
    if not s:
        return []
    result = []
    count = 1
    prev_char = s[0]

    for char in s[1:]:
        if char == prev_char:
            count += 1
        else:
            result.append((prev_char, count))
            prev_char = char
            count = 1
    result.append((prev_char, count))
    return result


def run_length_decoding(encoded_list: list[(str, int)]) -> str:
    """
    連長圧縮されたリストを復号して、元の文字列を生成します。

    引数:
        encoded_list (list[(str, int)]): 各文字とその連続出現回数のタプルからなるリスト。

    戻り値:
        str: 復号された元の文字列。

    使用例:
        >>> encoded_list = [('A', 4), ('B', 3), ('C', 2), ('D', 1), ('A', 2)]
        >>> original_string = run_length_decoding(encoded_list)
        >>> print(original_string)  # 出力: "AAAABBBCCDAA"
    """
    return ''.join(char * count for char, count in encoded_list)


# ============================================================================
def main():
    n, m = INN()
    s = [IS() for _ in range(n)]

    mf = MFGraph(n * m + 2)
    for i in range(n * m):
        r, c = divmod(i, m)
        if s[r][c] == ".":
            if (r + c) % 2 == 0:
                mf.add_edge(n * m, i, 1)
            else:
                mf.add_edge(i, n * m + 1, 1)

    to_idx = lambda rr, cc: rr * m + cc
    for i in range(n):
        for j in range(m):
            if (i + j) % 2 == 0 and s[i][j] == ".":
                frm = to_idx(i, j)
                for dx, dy in DIR4:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < n and 0 <= ny < m and s[nx][ny] == ".":
                        to_ = to_idx(nx, ny)
                        mf.add_edge(frm, to_, 1)

    cnt = mf.flow(n * m, n * m + 1)

    ans = [["."] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if s[i][j] == "#":
                ans[i][j] = "#"

    ei = mf.edges()
    for edge in ei:
        if edge.flow != 0 and edge.src != n * m and edge.dst != n * m + 1:
            r1, c1 = divmod(edge.src, m)
            r2, c2 = divmod(edge.dst, m)

            if r1 == r2:
                c_min, c_max = min(c1, c2), max(c1, c2)
                ans[r1][c_min] = ">"
                ans[r1][c_max] = "<"
            elif c1 == c2:
                r_min, r_max = min(r1, r2), max(r1, r2)
                ans[r_min][c1] = "v"
                ans[r_max][c1] = "^"

    print(cnt)
    for row in ans:
        print("".join(row))


if __name__ == '__main__':
    main()
