
from ipdb import set_trace
from typing import List

from itertools import combinations_with_replacement as cwr

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:

        combos = []
        rmax = int(target / min(candidates))
        for r in range(1, rmax + 1):
            for combo in cwr(candidates, r):
                if sum(combo) == target:
                    combos.append(combo)

        return combos

sol = Solution()
sol.combinationSum([2, 3, 6, 7], 7)
sol.combinationSum([2, 3, 5], 8)
sol.combinationSum([3, 6], 9)
sol.combinationSum([5, 10, 8, 4, 3, 12, 9], 27)
