
from ipdb import set_trace

class Solution:
    def climbStairs(self, n: int) -> int:

        # 1st step
        steps = {}
        steps[0] = 1
        steps[1] = 2

        # initialize counter for paths and enter loop
        paths = 0
        while True:

            # extract any steps that have met or exceeded n
            for step in list(steps.keys()):
                if steps[step] == n:
                    paths += 1
                    del steps[step]
                    continue
                if steps[step] > n:
                    del steps[step]

            sks = list(steps.keys())
            lsks = len(sks)
            if lsks == 0:
                break

            # add new steps
            for step in sks:
                steps[step + lsks] = steps[step] + 2
                steps[step] += 1

        return paths

sol = Solution()
assert sol.climbStairs(4) == 5
