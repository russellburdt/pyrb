
from ipdb import set_trace
from typing import List

from itertools import combinations

class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:

        # initialize
        candidates = sorted(candidates)
        csum = 0
        combos = []

        # find max size of a combo
        for rmax, candidate in enumerate(candidates):
            csum += candidate
            if csum > target:
                break
        if rmax == len(candidates) - 1:
            rmax += 1

        # store ...



        set_trace()

        for r in range(1, rmax + 1):
            for combo in combinations(candidates, r):
                if sum(combo) == target:
                    combos.append(combo)

        return [list(x) for x in set(combos)]

sol = Solution()
sol.combinationSum2([29,19,14,33,11,5,9,23,23,33,12,9,25,25,12,21,14,11,20,30,17,19,5,6,6,5,5,11,12,25,31,28,31,33,27,7,33,31,17,13,21,24,17,12,6,16,20,16,22,5], 28)
# sol.combinationSum2([1], 1)
# sol.combinationSum2([1, 1, 2], 8)
# sol.combinationSum2([10, 1, 2, 7, 6, 1, 5], 8)
