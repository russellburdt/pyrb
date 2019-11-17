
from ipdb import set_trace
from typing import List


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:

        candidates.sort()
        combos = []

        def csr(combo, index):
            print(combo, index)

            csum = sum(combo)
            if csum > target:
                return
            if csum == target:
                combos.append(combo)
                return
            csum = target - csum
            for x in range(index, len(candidates)):
                cx = candidates[x]
                if cx > csum:
                    break
                csr(combo + [cx], x)

        csr([], 0)
        return combos

sol = Solution()
sol.combinationSum([2, 3, 6, 7], 7)
# sol.combinationSum([2, 3, 5], 8)
# sol.combinationSum([3, 6], 9)
# sol.combinationSum([5, 10, 8, 4, 3, 12, 9], 27)



# from itertools import combinations_with_replacement as cwr
        # combos = []
        # rmax = int(target / min(candidates))
        # for r in range(1, rmax + 1):
        #     for combo in cwr(candidates, r):
        #         if sum(combo) == target:
        #             combos.append(combo)

        # return combos
