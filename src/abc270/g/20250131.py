import heapq
import math
import sys
from collections import defaultdict
from functools import lru_cache, cmp_to_key
from sortedcontainers import SortedList, SortedSet, SortedDict
from typing import Callable, TypeVar, Any, NamedTuple, Optional, cast

sys.setrecursionlimit(1000000)

#####################################################
# CONSTS
#####################################################
INF = 2 ** 60
MOD = 998244353
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
# String
#####################################################
# https://github.com/not522/ac-library-python/blob/master/atcoder/string.py
def _sa_naive(s: list[int]) -> list[int]:
    sa = list(range(len(s)))
    return sorted(sa, key=lambda i: s[i:])


def _sa_doubling(s: list[int]) -> list[int]:
    n = len(s)
    sa = list(range(n))
    rnk = s.copy()
    tmp = [0] * n
    k = 1
    while k < n:
        def cmp(x: int, y: int) -> int:
            if rnk[x] != rnk[y]:
                return rnk[x] - rnk[y]
            rx = rnk[x + k] if x + k < n else -1
            ry = rnk[y + k] if y + k < n else -1
            return rx - ry

        sa.sort(key=cmp_to_key(cmp))
        tmp[sa[0]] = 0
        for i in range(1, n):
            tmp[sa[i]] = tmp[sa[i - 1]] + (1 if cmp(sa[i - 1], sa[i]) else 0)
        tmp, rnk = rnk, tmp
        k *= 2
    return sa


def _sa_is(s: list[int], upper: int) -> list[int]:
    threshold_naive = 10
    threshold_doubling = 40

    n = len(s)

    if n == 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        if s[0] < s[1]:
            return [0, 1]
        else:
            return [1, 0]

    if n < threshold_naive:
        return _sa_naive(s)
    if n < threshold_doubling:
        return _sa_doubling(s)

    sa = [0] * n
    ls = [False] * n
    for i in range(n - 2, -1, -1):
        if s[i] == s[i + 1]:
            ls[i] = ls[i + 1]
        else:
            ls[i] = s[i] < s[i + 1]

    sum_l = [0] * (upper + 1)
    sum_s = [0] * (upper + 1)
    for i in range(n):
        if not ls[i]:
            sum_s[s[i]] += 1
        else:
            sum_l[s[i] + 1] += 1
    for i in range(upper + 1):
        sum_s[i] += sum_l[i]
        if i < upper:
            sum_l[i + 1] += sum_s[i]

    def induce(lms: list[int]) -> None:
        nonlocal sa
        sa = [-1] * n

        buf = sum_s.copy()
        for d in lms:
            if d == n:
                continue
            sa[buf[s[d]]] = d
            buf[s[d]] += 1

        buf = sum_l.copy()
        sa[buf[s[n - 1]]] = n - 1
        buf[s[n - 1]] += 1
        for i in range(n):
            v = sa[i]
            if v >= 1 and not ls[v - 1]:
                sa[buf[s[v - 1]]] = v - 1
                buf[s[v - 1]] += 1

        buf = sum_l.copy()
        for i in range(n - 1, -1, -1):
            v = sa[i]
            if v >= 1 and ls[v - 1]:
                buf[s[v - 1] + 1] -= 1
                sa[buf[s[v - 1] + 1]] = v - 1

    lms_map = [-1] * (n + 1)
    m = 0
    for i in range(1, n):
        if not ls[i - 1] and ls[i]:
            lms_map[i] = m
            m += 1
    lms = []
    for i in range(1, n):
        if not ls[i - 1] and ls[i]:
            lms.append(i)

    induce(lms)

    if m:
        sorted_lms = []
        for v in sa:
            if lms_map[v] != -1:
                sorted_lms.append(v)
        rec_s = [0] * m
        rec_upper = 0
        rec_s[lms_map[sorted_lms[0]]] = 0
        for i in range(1, m):
            left = sorted_lms[i - 1]
            right = sorted_lms[i]
            if lms_map[left] + 1 < m:
                end_l = lms[lms_map[left] + 1]
            else:
                end_l = n
            if lms_map[right] + 1 < m:
                end_r = lms[lms_map[right] + 1]
            else:
                end_r = n

            same = True
            if end_l - left != end_r - right:
                same = False
            else:
                while left < end_l:
                    if s[left] != s[right]:
                        break
                    left += 1
                    right += 1
                if left == n or s[left] != s[right]:
                    same = False

            if not same:
                rec_upper += 1
            rec_s[lms_map[sorted_lms[i]]] = rec_upper

        rec_sa = _sa_is(rec_s, rec_upper)

        for i in range(m):
            sorted_lms[i] = lms[rec_sa[i]]
        induce(sorted_lms)

    return sa


def suffix_array(s: str | list[int],
                 upper: Optional[int] = None) -> list[int]:
    """
    SA-IS による線形時間 Suffix Array (SA) 構築を行う。
    文字列 (str) または整数リスト (list[int]) を受け取り、その Suffix Array を返す。

    - 文字列の場合: 各文字を ord() で整数化し、最大値 255 (ASCII) を仮定して SA-IS を実行。
    - 整数リストの場合:
      - `upper` が指定されていれば、各要素の範囲は [0, upper] として SA-IS を実行。
      - `upper` が未指定の場合は、要素のユニーク値の昇順ランクを振ってから SA-IS を実行（要素を再マップ）。

    Args:
        s: 対象となる文字列または整数リスト。
        upper (Optional[int]): s が整数リストの場合、その最大値を指定することで高速化できる。
                               未指定なら自動でランク付けを行う。

    Returns:
        list[int]: Suffix Array。i 番目の要素が、辞書順で i 番目に来る接尾辞の開始インデックス。
    """

    if isinstance(s, str):
        return _sa_is([ord(c) for c in s], 255)
    elif upper is None:
        n = len(s)
        idx = list(range(n))

        def cmp(left: int, right: int) -> int:
            return cast(int, s[left]) - cast(int, s[right])

        idx.sort(key=cmp_to_key(cmp))
        s2 = [0] * n
        now = 0
        for i in range(n):
            if i and s[idx[i - 1]] != s[idx[i]]:
                now += 1
            s2[idx[i]] = now
        return _sa_is(s2, now)
    else:
        assert 0 <= upper
        for d in s:
            assert 0 <= d <= upper

        return _sa_is(s, upper)


def lcp_array(s: str | list[int],
              sa: list[int]) -> list[int]:
    """
    LCP (Longest Common Prefix) 配列を計算する関数。

    `s` の Suffix Array `sa` が既に与えられているとき、
    隣接する接尾辞同士の最長共通接頭辞の長さを、配列として返す。

    具体的には、`sa[i]` と `sa[i+1]` が指し示す接尾辞の LCP を計算し、それらを i 順に並べたもの。

    - 計算量は O(N)。

    Args:
        s: 文字列または整数リスト。文字列の場合は内部で ord(c) に変換。
        sa (list[int]): `s` の Suffix Array。長さ N。

    Returns:
        list[int]: 長さ N-1 の LCP 配列。LCP[i] は SA の i 番目と i+1 番目の接頭辞が共有する先頭一致長。
    """

    if isinstance(s, str):
        s = [ord(c) for c in s]

    n = len(s)
    assert n >= 1

    rnk = [0] * n
    for i in range(n):
        rnk[sa[i]] = i

    lcp = [0] * (n - 1)
    h = 0
    for i in range(n):
        if h > 0:
            h -= 1
        if rnk[i] == 0:
            continue
        j = sa[rnk[i] - 1]
        while j + h < n and i + h < n:
            if s[j + h] != s[i + h]:
                break
            h += 1
        lcp[rnk[i] - 1] = h

    return lcp


def z_algorithm(s: str | list[int]) -> list[int]:
    """
    Z-algorithm を用いて文字列 (または整数リスト) の Z-array を計算する。

    Z-array の定義：
    Z[i] は、文字列 s と s[i:] (接尾辞) が先頭からどれだけ最大で一致するかを示す。

    - Z[0] は文字列全体の長さ n が入ることに注意。
    - 時間計算量は O(n)。

    Args:
        s: 対象の文字列または整数リスト。文字列の場合は内部で ord(c) に変換される。

    Returns:
        list[int]: 長さ n の Z-array。Z[0] = n, i > 0 については s と s[i:] の先頭一致長。
    """

    if isinstance(s, str):
        s = [ord(c) for c in s]

    n = len(s)
    if n == 0:
        return []

    z = [0] * n
    j = 0
    for i in range(1, n):
        z[i] = 0 if j + z[j] <= i else min(j + z[j] - i, z[i - j])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if j + z[j] < i + z[i]:
            j = i
    z[0] = n

    return z


def rolling_hash(s: str, base: int = 100) -> tuple[list[int], list[int]]:
    """
    文字列 s に対するローリングハッシュと、base の累乗を計算して返す。
    **ROLLING_HASH_MOD** を使用することに注意。

    Args:
        s (str): ハッシュを取りたい文字列
        base (int, optional): ローリングハッシュを計算するときに使う基数. デフォルトは 100.

    Returns:
        tuple[list[int], list[int]]:
            - result: 長さ len(s) + 1 のリスト。result[i] は s[:i]（先頭 i 文字）のハッシュ値
            - bases:  長さ len(s) + 1 のリスト。bases[i] は base^i (mod ROLLING_HASH_MOD)
    """
    result = [0]
    bases = [1]
    for i in range(len(s)):
        crr = (result[-1] * base + ord(s[i]) + 1) % ROLLING_HASH_MOD
        result.append(crr)
        crr_base = (bases[-1] * base) % ROLLING_HASH_MOD
        bases.append(crr_base)
    return result, bases


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

    def debug_print(self) -> None:
        """
        Union-Find構造の内部状態を表示するデバッグ用関数。
        """
        print("=== Debug Print of UnionFind ===")
        print(f"Number of elements: {self.n}")
        print(f"parents: {self.parents}")
        print(f"roots: {self.roots}")
        print("All group members:")
        for root, members in self.all_group_members().items():
            print(f"  root {root} -> {members}")
        print("================================")


#####################################################
# SegTree
#####################################################
T = TypeVar('T')
"""
Tはセグメントツリーが扱う要素の型を表します。セグメントツリーは任意のデータ型に対して汎用的に使用できます。
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

    def debug_print(self) -> None:
        """
        内部構造をレベルごとに表示するデバッグ用関数。
        """
        print("=== Debug Print of Segment Tree ===")
        height = self.num.bit_length()
        for level in range(height + 1):
            start = 1 << level
            end = min((1 << (level + 1)), 2 * self.num)
            if start >= 2 * self.num:
                break

            nodes = []
            for i in range(start, end):
                nodes.append(f"{i}:{self.tree[i]}")
            print(f"Level {level} : " + " | ".join(nodes))
        print("===================================")


# https://github.com/not522/ac-library-python/blob/master/atcoder/lazysegtree.py
def _ceil_pow(n: int) -> int:
    x = 0
    while (1 << x) < n:
        x += 1

    return x


class LazySegTree:
    """
    遅延評価セグメント木 (Lazy Segment Tree) 。
    """

    def __init__(
            self,
            op: Callable[[Any, Any], Any],
            e: Any,
            mapping: Callable[[Any, Any], Any],
            composition: Callable[[Any, Any], Any],
            id_: Any,
            v: int | list[Any]) -> None:
        """
        コンストラクタ。必要な演算や単位元、遅延値の関数を受け取り、LazySegTree を初期化します。

        Args:
            op (Callable[[Any, Any], Any]): 区間をマージする演算関数
            e (Any): 演算 op における単位元
            mapping (Callable[[Any, Any], Any]): 遅延値を配列要素に適用する関数
            composition (Callable[[Any, Any], Any]): 遅延値どうしを合成する関数
            id_ (Any): 遅延値の単位元（「何もしない」ことを表す更新）
            v:
                - int の場合: サイズ v の配列をすべて e（単位元）で初期化
                - リストの場合: そのリストをもとにセグメント木を作成
        """
        self._op = op
        self._e = e
        self._mapping = mapping
        self._composition = composition
        self._id = id_

        if isinstance(v, int):
            v = [e] * v

        self._n = len(v)
        self._log = _ceil_pow(self._n)
        self._size = 1 << self._log
        self._d = [e] * (2 * self._size)
        self._lz = [self._id] * self._size
        for i in range(self._n):
            self._d[self._size + i] = v[i]
        for i in range(self._size - 1, 0, -1):
            self._update(i)

    def set(self, p: int, x: Any) -> None:
        """
        配列上の位置 p の値を x に直接書き換える (単一点更新)。

        Args:
            p (int): 更新する位置（0-indexed）
            x (Any): 更新後の値
        """
        assert 0 <= p < self._n

        p += self._size
        for i in range(self._log, 0, -1):
            self._push(p >> i)
        self._d[p] = x
        for i in range(1, self._log + 1):
            self._update(p >> i)

    def get(self, p: int) -> Any:
        """
        配列上の位置 p の値を取得する (単一点取得)。

        Args:
            p (int): 取得したい位置（0-indexed）

        Returns:
            Any: 位置 p の値
        """
        assert 0 <= p < self._n

        p += self._size
        for i in range(self._log, 0, -1):
            self._push(p >> i)
        return self._d[p]

    def query(self, left: int, right: int) -> Any:
        """
        区間 [left, right) に対する演算 `_op` の結果を取得する。

        Args:
            left (int): 演算を行う区間の開始インデックス (0-indexed, inclusive)
            right (int): 演算を行う区間の終了インデックス (0-indexed, exclusive)

        Returns:
            Any: 区間 [left, right) を演算 `_op` でまとめた結果
        """
        assert 0 <= left <= right <= self._n

        if left == right:
            return self._e

        left += self._size
        right += self._size

        for i in range(self._log, 0, -1):
            if ((left >> i) << i) != left:
                self._push(left >> i)
            if ((right >> i) << i) != right:
                self._push((right - 1) >> i)

        sml = self._e
        smr = self._e
        while left < right:
            if left & 1:
                sml = self._op(sml, self._d[left])
                left += 1
            if right & 1:
                right -= 1
                smr = self._op(self._d[right], smr)
            left >>= 1
            right >>= 1

        return self._op(sml, smr)

    def all_query(self) -> Any:
        """
        配列全体を演算 `_op` でまとめた結果を返す。

        Returns:
            Any: 全区間に対する演算結果
        """
        return self._d[1]

    def apply(self, left: int, right: Optional[int] = None,
              f: Optional[Any] = None) -> None:
        """
        区間更新または単一点更新を行う。

        (1) `apply(left, None, f)` の場合
            - 位置 `left` のみを `f` で更新（単一点更新）
        (2) `apply(left, right, f)` の場合
            - 区間 [left, right) に一括で `f` を適用（区間更新）

        Args:
            left (int): 更新を開始する位置（単一点の場合は更新対象位置）
            right (Optional[int]): 更新を終了する位置（exclusive）
                省略時は単一点更新
            f (Any): 適用する更新（遅延値）。None は不可

        Raises:
            AssertionError: `f` が None の場合や、引数が不正な場合
        """
        assert f is not None

        if right is None:
            p = left
            assert 0 <= left < self._n

            p += self._size
            for i in range(self._log, 0, -1):
                self._push(p >> i)
            self._d[p] = self._mapping(f, self._d[p])
            for i in range(1, self._log + 1):
                self._update(p >> i)
        else:
            assert 0 <= left <= right <= self._n
            if left == right:
                return

            left += self._size
            right += self._size

            for i in range(self._log, 0, -1):
                if ((left >> i) << i) != left:
                    self._push(left >> i)
                if ((right >> i) << i) != right:
                    self._push((right - 1) >> i)

            l2 = left
            r2 = right
            while left < right:
                if left & 1:
                    self._all_apply(left, f)
                    left += 1
                if right & 1:
                    right -= 1
                    self._all_apply(right, f)
                left >>= 1
                right >>= 1
            left = l2
            right = r2

            for i in range(1, self._log + 1):
                if ((left >> i) << i) != left:
                    self._update(left >> i)
                if ((right >> i) << i) != right:
                    self._update((right - 1) >> i)

    def max_right(
            self, left: int, g: Callable[[Any], bool]) -> int:
        """
        条件 g を満たす最大の右端インデックス right を求める。

        具体的には、[left, right) の区間について prod(left, x) を計算しながら
        g(...) が True である最大の x を二分探索のように探す。

        Args:
            left (int): 探索を開始する左端
            g (Callable[[Any], bool]): 条件判定関数。True の間は右を伸ばせる。

        Returns:
            int: [left, right) で g(...) が True になる最大の right
        """
        assert 0 <= left <= self._n
        assert g(self._e)

        if left == self._n:
            return self._n

        left += self._size
        for i in range(self._log, 0, -1):
            self._push(left >> i)

        sm = self._e
        first = True
        while first or (left & -left) != left:
            first = False
            while left % 2 == 0:
                left >>= 1
            if not g(self._op(sm, self._d[left])):
                while left < self._size:
                    self._push(left)
                    left *= 2
                    if g(self._op(sm, self._d[left])):
                        sm = self._op(sm, self._d[left])
                        left += 1
                return left - self._size
            sm = self._op(sm, self._d[left])
            left += 1

        return self._n

    def min_left(self, right: int, g: Any) -> int:
        """
        条件 g を満たす最小の左端インデックス left を求める。

        具体的には、[left, right) の区間について部分的に prod(x, right) を取りながら
        g(...) が True である最小の x を二分探索的に探す。

        Args:
            right (int): 探索を開始する右端
            g (Callable[[Any], bool]): 条件判定関数。True の間は左を詰められる。

        Returns:
            int: [left, right) で g(...) が True になる最小の left
        """
        assert 0 <= right <= self._n
        assert g(self._e)

        if right == 0:
            return 0

        right += self._size
        for i in range(self._log, 0, -1):
            self._push((right - 1) >> i)

        sm = self._e
        first = True
        while first or (right & -right) != right:
            first = False
            right -= 1
            while right > 1 and right % 2:
                right >>= 1
            if not g(self._op(self._d[right], sm)):
                while right < self._size:
                    self._push(right)
                    right = 2 * right + 1
                    if g(self._op(self._d[right], sm)):
                        sm = self._op(self._d[right], sm)
                        right -= 1
                return right + 1 - self._size
            sm = self._op(self._d[right], sm)

        return 0

    def _update(self, k: int) -> None:
        self._d[k] = self._op(self._d[2 * k], self._d[2 * k + 1])

    def _all_apply(self, k: int, f: Any) -> None:
        self._d[k] = self._mapping(f, self._d[k])
        if k < self._size:
            self._lz[k] = self._composition(f, self._lz[k])

    def _push(self, k: int) -> None:
        self._all_apply(2 * k, self._lz[k])
        self._all_apply(2 * k + 1, self._lz[k])
        self._lz[k] = self._id

    def debug_print(self) -> None:
        """
        Lazy Segment Tree の内部構造 (_d と _lz) をレベルごとに表示するデバッグ用関数。
        """
        print("=== Debug Print of LazySegTree ===")

        print("[_d array in level order]")
        height_d = self._size.bit_length()
        for level in range(height_d):
            start = 1 << level
            end = min((1 << (level + 1)), 2 * self._size)
            if start >= 2 * self._size:
                break

            nodes = []
            for i in range(start, end):
                nodes.append(f"{i}:{self._d[i]}")
            print(f"Level {level}: " + " | ".join(nodes))

        print("\n[_lz array in level order]")
        height_lz = self._size.bit_length()
        for level in range(height_lz):
            start = 1 << level
            end = min((1 << (level + 1)), self._size)
            if start >= self._size:
                break

            nodes = []
            for i in range(start, end):
                nodes.append(f"{i}:{self._lz[i]}")
            print(f"Lazy Level {level}: " + " | ".join(nodes))

        print("===================================")


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
        array (list[int]): 転倒数を計算したい整数の配列。

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
# https://github.com/not522/ac-library-python/blob/master/atcoder/maxflow.py


class MFGraph:
    """
    最大流を求めるグラフクラス。

    ノードは 0 ～ n-1 の範囲で扱い、内部では各ノードの隣接リストを _g に保持する。
    """

    class Edge(NamedTuple):
        """
        公開用のエッジ情報を格納する NamedTuple。

        Attributes:
            src (int): エッジの始点ノード。
            dst (int): エッジの終点ノード。
            cap (int): エッジの総容量（元のエッジと逆向きエッジの和）。
            flow (int): 実際に流れているフロー量（逆向きエッジの容量側）。
        """
        src: int
        dst: int
        cap: int
        flow: int

    class _Edge:
        """
        内部で使用するエッジ情報を格納するクラス。

        Attributes:
            dst (int): 接続先ノード。
            cap (int): 残余グラフでの容量。
            rev (Optional[MFGraph._Edge]): 逆向きエッジへの参照。
        """

        def __init__(self, dst: int, cap: int) -> None:
            self.dst = dst
            self.cap = cap
            self.rev: Optional[MFGraph._Edge] = None

    def __init__(self, n: int) -> None:
        """
        MFGraphの初期化メソッド。

        ノード数nを指定してグラフを初期化する。0～n-1 のノードを扱う。

        パラメータ:
            n (int): ノード数。
        """
        self._n = n
        self._g: list[list[MFGraph._Edge]] = [[] for _ in range(n)]
        self._edges: list[MFGraph._Edge] = []

    def add_edge(self, src: int, dst: int, cap: int) -> int:
        """
        グラフに有向エッジ(src -> dst)を容量capで追加し、対応する逆向きエッジ(dst -> src)の容量を0で追加する。

        パラメータ:
            src (int): エッジの始点ノード。
            dst (int): エッジの終点ノード。
            cap (int): エッジの容量。

        戻り値:
            int: 登録されたエッジのインデックス番号。get_edge() 等で取得するときに使用する。
        """
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
        """
        add_edgeで追加した i 番目のエッジに対応する情報を取得する。

        パラメータ:
            i (int): add_edge() で返されたエッジのインデックス。

        戻り値:
            MFGraph.Edge: (src, dst, cap, flow) の4つを持つ NamedTuple。
                          src -> dst の元エッジと逆向きエッジの容量、フロー量を反映。
        """
        assert 0 <= i < len(self._edges)
        e = self._edges[i]
        re = cast(MFGraph._Edge, e.rev)
        return MFGraph.Edge(
            re.dst,
            e.dst,
            e.cap + re.cap,
            re.cap
        )

    def edges(self) -> list[Edge]:
        """
        グラフに登録されているすべてのエッジの情報をリストで取得する。

        戻り値:
            list[MFGraph.Edge]: get_edge(i) をすべて i について呼んだ結果を返す。
        """
        return [self.get_edge(i) for i in range(len(self._edges))]

    def change_edge(self, i: int, new_cap: int, new_flow: int) -> None:
        """
        既存の i 番目のエッジ容量とフロー量を変更する。

        パラメータ:
            i (int): 変更対象のエッジインデックス。
            new_cap (int): 新しい容量。
            new_flow (int): 新しいフロー量。0 <= new_flow <= new_cap を満たす必要がある。
        """
        assert 0 <= i < len(self._edges)
        assert 0 <= new_flow <= new_cap
        e = self._edges[i]
        e.cap = new_cap - new_flow
        assert e.rev is not None
        e.rev.cap = new_flow

    def flow(self, s: int, t: int, flow_limit: Optional[int] = None) -> int:
        """
        s から t へ、与えられた flow_limit を上限としてフローを流す。

        レベルグラフを構築したうえで、DFS または BFS を組み合わせる方式で可能な限りフローを流す。

        パラメータ:
            s (int): フローを流し始めるソースノード。
            t (int): フローを受け取るシンクノード。
            flow_limit (Optional[int]): フローの上限。指定しない場合はソースから出るエッジ容量の合計が上限となる。

        戻り値:
            int: 実際に流れたフロー量。
        """
        assert 0 <= s < self._n
        assert 0 <= t < self._n
        assert s != t
        if flow_limit is None:
            flow_limit = cast(int, sum(e.cap for e in self._g[s]))

        current_edge = [0] * self._n
        level = [0] * self._n

        def fill(arr: list[int], value: int) -> None:
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
            edge_stack: list[MFGraph._Edge] = []
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

    def min_cut(self, s: int) -> list[bool]:
        """
        s から到達可能な頂点集合を探し、最小カットを示す部分集合を返す。

        max_flow 後の残余グラフで、s からたどり着けるノードを True、
        たどり着けないノードを False として返す。

        パラメータ:
            s (int): 始点ノード。

        戻り値:
            list[bool]: 各ノードが s から到達可能かどうかを表すブール値のリスト。
        """
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
# Min Cost Flow
#####################################################
# https://github.com/not522/ac-library-python/blob/master/atcoder/mincostflow.py
class MCFGraph:
    """
    最小費用流 (Min-Cost Flow) を扱うグラフクラス。

    頂点数 `n` の有向グラフに容量付き・コスト付きの有向辺を追加し、
    ソース `s` からシンク `t` への最大流量とそのときの最小費用を求めることができる。

    Attributes:
        _n (int): 頂点数。
        _g (list[list[MCFGraph._Edge]]): 各頂点が持つ有向辺のリスト。
        _edges (list[MCFGraph._Edge]): 追加した辺を記憶するためのリスト。
    """

    class Edge(NamedTuple):
        """
        公開用のエッジ情報を表す NamedTuple。

        Attributes:
            src (int): エッジの始点ノード。
            dst (int): エッジの終点ノード。
            cap (int): エッジの容量（src->dst）。
            flow (int): 現在流れているフロー量。
            cost (int): エッジ1単位あたりのコスト。
        """
        src: int
        dst: int
        cap: int
        flow: int
        cost: int

    class _Edge:
        def __init__(self, dst: int, cap: int, cost: int) -> None:
            self.dst = dst
            self.cap = cap
            self.cost = cost
            self.rev: Optional[MCFGraph._Edge] = None

    def __init__(self, n: int) -> None:
        """
        コンストラクタ。

        Args:
            n (int): 頂点数。0 から n-1 の頂点を扱う。
        """
        self._n = n
        self._g: list[list[MCFGraph._Edge]] = [[] for _ in range(n)]
        self._edges: list[MCFGraph._Edge] = []

    def add_edge(self, src: int, dst: int, cap: int, cost: int) -> int:
        """
        グラフに容量 `cap`、コスト `cost` の有向辺 (src -> dst) を追加し、逆向き辺は容量0、コスト -cost で追加する。

        Args:
            src (int): エッジの始点ノード。0 <= src < n
            dst (int): エッジの終点ノード。0 <= dst < n
            cap (int): このエッジが持つ容量。非負整数。
            cost (int): エッジ1単位あたりのコスト。整数（負でも可）。

        Returns:
            int: 登録されたエッジのインデックス。get_edge() で参照するときに使用。
        """
        assert 0 <= src < self._n
        assert 0 <= dst < self._n
        assert 0 <= cap
        m = len(self._edges)
        e = MCFGraph._Edge(dst, cap, cost)
        re = MCFGraph._Edge(src, 0, -cost)
        e.rev = re
        re.rev = e
        self._g[src].append(e)
        self._g[dst].append(re)
        self._edges.append(e)
        return m

    def get_edge(self, i: int) -> Edge:
        """
        add_edge で追加した i 番目のエッジ情報を取得する。

        具体的には src, dst, cap(元々の容量), flow(流れている量), cost(コスト) をまとめた
        NamedTuple (MCFGraph.Edge) を返す。

        Args:
            i (int): 取得するエッジのインデックス。add_edge() の戻り値。

        Returns:
            MCFGraph.Edge: i 番目のエッジ情報。

        Raises:
            AssertionError: i が範囲外の場合。
        """
        assert 0 <= i < len(self._edges)
        e = self._edges[i]
        re = cast(MCFGraph._Edge, e.rev)
        return MCFGraph.Edge(
            re.dst,
            e.dst,
            e.cap + re.cap,
            re.cap,
            e.cost
        )

    def edges(self) -> list[Edge]:
        """
        追加したすべてのエッジ情報を取得する。

        Returns:
            list[MCFGraph.Edge]: 登録済みの全エッジを表す Edge のリスト。
        """
        return [self.get_edge(i) for i in range(len(self._edges))]

    def flow(self, s: int, t: int,
             flow_limit: Optional[int] = None) -> tuple[int, int]:
        """
        頂点 s から頂点 t へ、与えられた flow_limit を上限としてフローを流し、
        (最大流量, 最小費用) のペアを返す。

        Args:
            s (int): ソース頂点 (流量の始点)。
            t (int): シンク頂点 (流量の終点)。
            flow_limit (Optional[int]): 流量の上限。指定しない場合は s から出る辺容量の合計が上限となる。

        Returns:
            tuple[int, int]:
                - 0番目: 実際に流せたフロー量
                - 1番目: そのときの合計コスト
        """
        return self.slope(s, t, flow_limit)[-1]

    def slope(self, s: int, t: int,
              flow_limit: Optional[int] = None) -> list[tuple[int, int]]:
        """
        フロー量を少しずつ増やしながら、(フロー量, コスト) のペアを生成して返す。

        - フロー量を 0 から flow_limit まで単調に増やしていく過程での
          (累積流量, 累積コスト) の各ポイントをリストとして返す。

        Args:
            s (int): ソース頂点 (流量の始点)。
            t (int): シンク頂点 (流量の終点)。
            flow_limit (Optional[int]): 流量の上限。指定しない場合は s から出る辺容量の合計が上限となる。

        Returns:
            list[tuple[int, int]]:
                各ステップごとの (累積流量, 累積コスト) のペアの一覧。
                例: [(0, 0), (f1, c1), (f2, c2), ... (F, C)] のような形。
                ここで F, C は最終的なフロー量とコスト。
        """
        assert 0 <= s < self._n
        assert 0 <= t < self._n
        assert s != t
        if flow_limit is None:
            flow_limit = cast(int, sum(e.cap for e in self._g[s]))

        dual = [0] * self._n
        prev: list[Optional[tuple[int, MCFGraph._Edge]]] = [None] * self._n

        def refine_dual() -> bool:
            pq = [(0, s)]
            visited = [False] * self._n
            dist: list[Optional[int]] = [None] * self._n
            dist[s] = 0
            while pq:
                dist_v, v = heapq.heappop(pq)
                if visited[v]:
                    continue
                visited[v] = True
                if v == t:
                    break
                dual_v = dual[v]
                for e in self._g[v]:
                    w = e.dst
                    if visited[w] or e.cap == 0:
                        continue
                    reduced_cost = e.cost - dual[w] + dual_v
                    new_dist = dist_v + reduced_cost
                    dist_w = dist[w]
                    if dist_w is None or new_dist < dist_w:
                        dist[w] = new_dist
                        prev[w] = v, e
                        heapq.heappush(pq, (new_dist, w))
            else:
                return False
            dist_t = dist[t]
            for v in range(self._n):
                if visited[v]:
                    dual[v] -= cast(int, dist_t) - cast(int, dist[v])
            return True

        flow = 0
        cost = 0
        prev_cost_per_flow: Optional[int] = None
        result = [(flow, cost)]
        while flow < flow_limit:
            if not refine_dual():
                break
            f = flow_limit - flow
            v = t
            while prev[v] is not None:
                u, e = cast(tuple[int, MCFGraph._Edge], prev[v])
                f = min(f, e.cap)
                v = u
            v = t
            while prev[v] is not None:
                u, e = cast(tuple[int, MCFGraph._Edge], prev[v])
                e.cap -= f
                assert e.rev is not None
                e.rev.cap += f
                v = u
            c = -dual[s]
            flow += f
            cost += f * c
            if c == prev_cost_per_flow:
                result.pop()
            result.append((flow, cost))
            prev_cost_per_flow = c
        return result


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
        int | list[int]:
            - `goal` が指定された場合は、開始ノードから `goal` ノードへの最短距離を返します。
              ただし、到達不能な場合は-1を返します。
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
    return -1 if goal is not None else dists1


def floyd_warshall(n: int, paths: list[list[tuple[int, int]]]) -> list[list[int]]:
    """
    ワーシャルフロイド法を用いて、全ノード間の最短距離を求めます。

    Args:
        n (int): グラフのノード数。ノードは0からn-1までの整数で表されます。
        paths (list[list[tuple[int, int]]]):
            各ノードから接続されているノードとその距離のリスト。
            例えば、paths[u] に (v, w) が含まれている場合、
            ノードuからノードvへの距離はwとなります。

    Returns:
        list[list[int]]:
            ノードiからノードjへの最短距離を dist[i][j] とした二次元リストを返します。
            到達不可能な場合は -1 が設定されます。
    """
    dist = [[INF] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0

    for u in range(n):
        for v, w in paths[u]:
            dist[u][v] = min(dist[u][v], w)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    for i in range(n):
        for j in range(n):
            if dist[i][j] == INF:
                dist[i][j] = -1

    return dist


class _CSR:
    def __init__(
            self, n: int, edges: list[tuple[int, int]]) -> None:
        self.start = [0] * (n + 1)
        self.elist = [0] * len(edges)

        for e in edges:
            self.start[e[0] + 1] += 1

        for i in range(1, n + 1):
            self.start[i] += self.start[i - 1]

        counter = self.start.copy()
        for e in edges:
            self.elist[counter[e[0]]] = e[1]
            counter[e[0]] += 1


class _SCCGraph:
    def __init__(self, n: int) -> None:
        self._n = n
        self._edges: list[tuple[int, int]] = []

    def num_vertices(self) -> int:
        return self._n

    def add_edge(self, from_vertex: int, to_vertex: int) -> None:
        self._edges.append((from_vertex, to_vertex))

    def scc_ids(self) -> tuple[int, list[int]]:
        g = _CSR(self._n, self._edges)
        now_ord = 0
        group_num = 0
        visited = []
        low = [0] * self._n
        order = [-1] * self._n
        ids = [0] * self._n

        sys.setrecursionlimit(max(self._n + 1000, sys.getrecursionlimit()))

        def dfs(v: int) -> None:
            nonlocal now_ord
            nonlocal group_num
            nonlocal visited
            nonlocal low
            nonlocal order
            nonlocal ids

            low[v] = now_ord
            order[v] = now_ord
            now_ord += 1
            visited.append(v)
            for i in range(g.start[v], g.start[v + 1]):
                to = g.elist[i]
                if order[to] == -1:
                    dfs(to)
                    low[v] = min(low[v], low[to])
                else:
                    low[v] = min(low[v], order[to])

            if low[v] == order[v]:
                while True:
                    u = visited[-1]
                    visited.pop()
                    order[u] = self._n
                    ids[u] = group_num
                    if u == v:
                        break
                group_num += 1

        for i in range(self._n):
            if order[i] == -1:
                dfs(i)

        for i in range(self._n):
            ids[i] = group_num - 1 - ids[i]

        return group_num, ids

    def scc(self) -> list[list[int]]:
        ids = self.scc_ids()
        group_num = ids[0]
        counts = [0] * group_num
        for x in ids[1]:
            counts[x] += 1
        groups: list[list[int]] = [[] for _ in range(group_num)]
        for i in range(self._n):
            groups[ids[1][i]].append(i)

        return groups


class SCCGraph:
    """
    強連結成分分解 (SCC: Strongly Connected Components) を扱うクラス。

    Example:
        >>> g = SCCGraph(5)
        >>> g.add_edge(0, 1)
        >>> g.add_edge(1, 2)
        >>> g.add_edge(2, 0)
        >>> g.add_edge(2, 3)
        >>> g.add_edge(3, 4)
        >>> sccs = g.scc()
        >>> print(sccs)  # 例: [[0, 1, 2], [3], [4]]
    """

    def __init__(self, n: int = 0) -> None:
        """
        コンストラクタ。

        n 個の頂点を持つ有向グラフを初期化する。
        頂点は 0 から n-1 までを扱う。

        Args:
            n (int, optional): グラフの頂点数。デフォルトは 0。
        """
        self._internal = _SCCGraph(n)

    def add_edge(self, from_vertex: int, to_vertex: int) -> None:
        """
        グラフに有向辺を追加する。

        Args:
            from_vertex (int): 辺の始点となる頂点番号 (0 <= from_vertex < n)。
            to_vertex (int): 辺の終点となる頂点番号 (0 <= to_vertex < n)。

        Raises:
            AssertionError: from_vertex, to_vertex の値が頂点範囲外のとき。
        """
        n = self._internal.num_vertices()
        assert 0 <= from_vertex < n
        assert 0 <= to_vertex < n
        self._internal.add_edge(from_vertex, to_vertex)

    def scc(self) -> list[list[int]]:
        """
        強連結成分分解を行い、その結果を返す。

        Returns:
            list[list[int]]:
                グラフを構成する強連結成分のリスト。
                各強連結成分はその頂点番号のリストとして表される。

        Note:
            AtCoder libraryではトポロジカルソートされており、u から v に到達できる時、u の属するリストは v の属するリストより前に登場する。
        """
        return self._internal.scc()


#####################################################
# 2-SAT
#####################################################
# https://github.com/not522/ac-library-python/blob/master/atcoder/twosat.py
class TwoSAT:
    """
    2-SAT (2-Satisfiability) 問題を扱うクラス。

    与えられた n 個のブール変数 x_0, x_1, ..., x_(n-1) について、
    それぞれ真 (True) / 偽 (False) の割り当てを探す問題を解く。
    """

    def __init__(self, n: int = 0) -> None:
        """
        コンストラクタ。n 個の変数を持つ 2-SAT インスタンスを初期化する。

        Args:
            n (int, optional): ブール変数の個数。デフォルトは 0。
        """
        self._n = n
        self._answer = [False] * n
        self._scc = _SCCGraph(2 * n)

    def add_clause(self, i: int, f: bool, j: int, g: bool) -> None:
        """
        節 (clause) を追加する。形としては (x_i = f) → (x_j = g) および (x_j = g) → (x_i = f)
        に相当する含意をグラフに追加する。

        Args:
            i (int): 変数のインデックス (0 <= i < n)
            f (bool): 変数 x_i を True/False のどちらとみなすか
            j (int): 変数のインデックス (0 <= j < n)
            g (bool): 変数 x_j を True/False のどちらとみなすか

        Raises:
            AssertionError: i, j が変数の範囲外のとき
        """
        assert 0 <= i < self._n
        assert 0 <= j < self._n

        self._scc.add_edge(2 * i + (0 if f else 1), 2 * j + (1 if g else 0))
        self._scc.add_edge(2 * j + (0 if g else 1), 2 * i + (1 if f else 0))

    def satisfiable(self) -> bool:
        """
        これまでに追加した節 (clause) をすべて満たす割り当てが存在するかを判定し、
        同時に割り当て結果を内部に記録する。

        Returns:
            bool: 充足可能なら True、充足不可能なら False。
                  充足可能な場合、`self._answer` に解を保存する。
        """
        scc_id = self._scc.scc_ids()[1]
        for i in range(self._n):
            if scc_id[2 * i] == scc_id[2 * i + 1]:
                return False
            self._answer[i] = scc_id[2 * i] < scc_id[2 * i + 1]
        return True

    def answer(self) -> list[bool]:
        """
        satisfiable() が True を返したときに確定した各変数の割り当てを返す。

        Returns:
            list[bool]: n 個の真偽値の配列。i 番目が変数 x_i の値 (True or False) となる。
        """
        return self._answer


#####################################################
# Matrix
#####################################################
def rotate_matrix(matrix: list[list[any]] | list[str], n: int) -> list[list[any]]:
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


def transpose_matrix(matrix: list[list[any]] | list[str]) -> list[list[any]]:
    """
    n行m列の行列の転置行列を返す関数

    Args:
        matrix: 転置の対象となる行列
    Returns:
        list[list[any]]: matrix の転置行列
    """
    return [list(row) for row in zip(*matrix)]


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


#####################################################
# Convolution
#####################################################
# https://github.com/shakayami/ACL-for-python/blob/master/convolution.py
class FFT:
    @staticmethod
    def _primitive_root_constexpr(m: int) -> int:
        """
        与えられた法 m に対して、最小の原始根を求める。

        :param m: 法 (mod)
        :return: m の最小原始根
        """
        if m == 2: return 1
        if m == 167772161: return 3
        if m == 469762049: return 3
        if m == 754974721: return 11
        if m == 998244353: return 3
        divs = [0] * 20
        divs[0] = 2
        cnt = 1
        x = (m - 1) // 2
        while x % 2 == 0: x //= 2
        i = 3
        while i * i <= x:
            if x % i == 0:
                divs[cnt] = i
                cnt += 1
                while x % i == 0:
                    x //= i
            i += 2
        if x > 1:
            divs[cnt] = x
            cnt += 1
        g = 2
        while 1:
            ok = True
            for i in range(cnt):
                if pow(g, (m - 1) // divs[i], m) == 1:
                    ok = False
                    break
            if ok:
                return g
            g += 1

    @staticmethod
    def _bfs(x: int) -> int:
        """
        与えられた整数 x が 2 で何回割り切れるか (2 の因数の数) を求める。

        :param x: 整数
        :return: x が 2 で割り切れる回数 (x を素因数分解したときの 2 の指数)
        """
        res = 0
        while x % 2 == 0:
            res += 1
            x //= 2
        return res

    rank2 = 0
    root = []
    iroot = []
    rate2 = []
    irate2 = []
    rate3 = []
    irate3 = []

    def __init__(self, mod: int):
        """
        :param mod: FFT を行う法 (素数)
        """
        self.mod = mod
        self.g = self._primitive_root_constexpr(self.mod)
        self.rank2 = self._bfs(self.mod - 1)
        self.root = [0 for _ in range(self.rank2 + 1)]
        self.iroot = [0 for _ in range(self.rank2 + 1)]
        self.rate2 = [0 for _ in range(self.rank2)]
        self.irate2 = [0 for _ in range(self.rank2)]
        self.rate3 = [0 for _ in range(self.rank2 - 1)]
        self.irate3 = [0 for _ in range(self.rank2 - 1)]
        self.root[self.rank2] = pow(self.g, (self.mod - 1) >> self.rank2, self.mod)
        self.iroot[self.rank2] = pow(self.root[self.rank2], self.mod - 2, self.mod)
        for i in range(self.rank2 - 1, -1, -1):
            self.root[i] = (self.root[i + 1] ** 2) % self.mod
            self.iroot[i] = (self.iroot[i + 1] ** 2) % self.mod
        prod = 1
        iprod = 1
        for i in range(self.rank2 - 1):
            self.rate2[i] = (self.root[i + 2] * prod) % self.mod
            self.irate2[i] = (self.iroot[i + 2] * iprod) % self.mod
            prod = (prod * self.iroot[i + 2]) % self.mod
            iprod = (iprod * self.root[i + 2]) % self.mod
        prod = 1
        iprod = 1
        for i in range(self.rank2 - 2):
            self.rate3[i] = (self.root[i + 3] * prod) % self.mod
            self.irate3[i] = (self.iroot[i + 3] * iprod) % self.mod
            prod = (prod * self.iroot[i + 3]) % self.mod
            iprod = (iprod * self.root[i + 3]) % self.mod

    def butterfly(self, a: list[int]) -> None:
        """
        配列 a に対して in-place にバタフライ演算を行う。

        :param a: 長さが 2^k の配列 (要素は法 self.mod 下での値)
        :return: 変換後、配列 a は FFT の途中段階の値に上書きされる
        """
        n = len(a)
        h = (n - 1).bit_length()

        LEN = 0
        while LEN < h:
            if h - LEN == 1:
                p = 1 << (h - LEN - 1)
                rot = 1
                for s in range(1 << LEN):
                    offset = s << (h - LEN)
                    for i in range(p):
                        l = a[i + offset]
                        r = a[i + offset + p] * rot
                        a[i + offset] = (l + r) % self.mod
                        a[i + offset + p] = (l - r) % self.mod
                    rot *= self.rate2[(~s & -~s).bit_length() - 1]
                    rot %= self.mod
                LEN += 1
            else:
                p = 1 << (h - LEN - 2)
                rot = 1
                imag = self.root[2]
                for s in range(1 << LEN):
                    rot2 = (rot * rot) % self.mod
                    rot3 = (rot2 * rot) % self.mod
                    offset = s << (h - LEN)
                    for i in range(p):
                        a0 = a[i + offset]
                        a1 = a[i + offset + p] * rot
                        a2 = a[i + offset + 2 * p] * rot2
                        a3 = a[i + offset + 3 * p] * rot3
                        a1na3imag = (a1 - a3) % self.mod * imag
                        a[i + offset] = (a0 + a2 + a1 + a3) % self.mod
                        a[i + offset + p] = (a0 + a2 - a1 - a3) % self.mod
                        a[i + offset + 2 * p] = (a0 - a2 + a1na3imag) % self.mod
                        a[i + offset + 3 * p] = (a0 - a2 - a1na3imag) % self.mod
                    rot *= self.rate3[(~s & -~s).bit_length() - 1]
                    rot %= self.mod
                LEN += 2

    def butterfly_inv(self, a: list[int]) -> None:
        """
        配列 a に対して in-place に逆バタフライ演算を行う。

        :param a: 長さが 2^k の配列 (要素は法 self.mod 下での値)
        :return: 変換後、配列 a は逆 FFT の途中段階の値に上書きされる
        """
        n = len(a)
        h = (n - 1).bit_length()
        LEN = h
        while LEN:
            if LEN == 1:
                p = 1 << (h - LEN)
                irot = 1
                for s in range(1 << (LEN - 1)):
                    offset = s << (h - LEN + 1)
                    for i in range(p):
                        l = a[i + offset]
                        r = a[i + offset + p]
                        a[i + offset] = (l + r) % self.mod
                        a[i + offset + p] = (l - r) * irot % self.mod
                    irot *= self.irate2[(~s & -~s).bit_length() - 1]
                    irot %= self.mod
                LEN -= 1
            else:
                p = 1 << (h - LEN)
                irot = 1
                iimag = self.iroot[2]
                for s in range(1 << (LEN - 2)):
                    irot2 = (irot * irot) % self.mod
                    irot3 = (irot * irot2) % self.mod
                    offset = s << (h - LEN + 2)
                    for i in range(p):
                        a0 = a[i + offset]
                        a1 = a[i + offset + p]
                        a2 = a[i + offset + 2 * p]
                        a3 = a[i + offset + 3 * p]
                        a2na3iimag = (a2 - a3) * iimag % self.mod
                        a[i + offset] = (a0 + a1 + a2 + a3) % self.mod
                        a[i + offset + p] = (a0 - a1 + a2na3iimag) * irot % self.mod
                        a[i + offset + 2 * p] = (a0 + a1 - a2 - a3) * irot2 % self.mod
                        a[i + offset + 3 * p] = (a0 - a1 - a2na3iimag) * irot3 % self.mod
                    irot *= self.irate3[(~s & -~s).bit_length() - 1]
                    irot %= self.mod
                LEN -= 2

    def convolution(self, a: list[int], b: list[int]) -> list[int]:
        """
        2 つの配列 a, b の畳み込み演算を法 self.mod の下で行い、その結果を返す。

        高速畳み込み (FFT) を利用し、O(n log n) (n は畳み込み後の配列長) で計算する。
        ただし配列の長さが小さい場合 (min(n, m) <= 40) は O(nm) の直接計算を行う。

        :param a: 法 self.mod 下で扱う整数列
        :param b: 法 self.mod 下で扱う整数列
        :return: a と b の畳み込み結果 (長さ len(a) + len(b) - 1 のリスト)
        """
        n = len(a)
        m = len(b)
        if not a or not b:
            return []
        if min(n, m) <= 40:
            res = [0] * (n + m - 1)
            for i in range(n):
                for j in range(m):
                    res[i + j] += a[i] * b[j]
                    res[i + j] %= self.mod
            return res
        z = 1 << ((n + m - 2).bit_length())
        a = a + [0] * (z - n)
        b = b + [0] * (z - m)
        self.butterfly(a)
        self.butterfly(b)
        c = [(a[i] * b[i]) % self.mod for i in range(z)]
        self.butterfly_inv(c)
        iz = pow(z, self.mod - 2, self.mod)
        for i in range(n + m - 1):
            c[i] = (c[i] * iz) % self.mod
        return c[:n + m - 1]


# ============================================================================
def main():
    def solve(P, A, B, S, G):
        if S == G:
            return 0
        if A == 0:
            if B == G:
                return 1
            else:
                return -1

        def f(X):
            return (A * X + B) % P

        # ステップ数
        M = math.ceil(math.sqrt(P))

        inv_A = pow(A, -1, P)
        c = (- B * inv_A) % P  # (-B) / A (mod P)

        C, D = 1, 0
        for _ in range(M):
            C = (inv_A * C) % P
            D = (inv_A * D + c) % P

        def f_inv_m(X):
            return (C * X + D) % P

        # Baby-step
        table = dict()
        X = S
        for j in range(M):
            if X not in table:
                table[X] = j
            X = f(X)

        # Giant-step
        crr = G
        for i in range(M):
            if crr in table:
                return i * M + table[crr]
            crr = f_inv_m(crr)

        return -1

    t = IN()
    for _ in range(t):
        p, a, b, s, g = INN()
        print(solve(p, a, b, s, g))
    return


if __name__ == '__main__':
    main()
