
from typing import List
from ipdb import set_trace

import numpy as np

class Solution:
    def maxProfitAssignment(self, difficulty: List[int], profit: List[int], worker: List[int]) -> int:

        mp = 0
        pdict = {}
        idx = np.argsort(difficulty)
        difficulty = np.array(difficulty)[idx]
        profit = np.array(profit)[idx]

        for wr in worker:
            if wr in pdict:
                mp += pdict[wr]
            else:
                idx = difficulty <= wr
                if np.any(idx):
                    pdict[wr] = profit[idx].max()
                    mp += pdict[wr]
                else:
                    pdict[wr] = 0
        return mp


sol = Solution()
assert sol.maxProfitAssignment(difficulty=[13, 37, 58], profit=[4, 90, 96], worker=[34, 73, 45]) == 190
assert sol.maxProfitAssignment(difficulty=[2, 4, 6, 8, 10], profit=[10, 20, 30, 40, 50], worker=[4, 5, 6, 7]) == 100
assert sol.maxProfitAssignment(difficulty=[85, 47, 57], profit=[24, 66, 99], worker=[40, 25, 25]) == 0
