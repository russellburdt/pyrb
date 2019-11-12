
from ipdb import set_trace
from typing import List

from itertools import combinations

class Solution:
    def maxProduct(self, words: List[str]) -> int:
        pmax = 0
        words = [''.join(list(set(x))) for x in words]
        for xa, xb in combinations(words, 2):
            ok = True
            for letter in xa:
                if letter in xb:
                    ok = False
                    break
            if ok:
                pmax = max(pmax, len(xa) * len(xb))
        return pmax


sol = Solution()
assert sol.maxProduct(words=["abcw","baz","foo","bar","xtfn","abcdef"]) == 16
assert sol.maxProduct(words=["a","ab","abc","d","cd","bcd","abcd"]) == 4
assert sol.maxProduct(words=["a","aa","aaa","aaaa"]) == 0
