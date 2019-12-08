
from ipdb import set_trace

class Solution:
    def countSubstrings(self, s: str) -> int:

        # container for palindrome indices
        pd = {}

        # identify range objects representing palindromes indices of length 1 and 2
        pd[1] = [range(x, x + 1) for x in range(len(s))]
        pd[2] = []
        for x in range(len(s) - 1):
            if s[x] == s[x + 1]:
                pd[2].append(range(x, x + 2))

        # fill in palindromes of greater complexity
        idx = 3
        more = True
        while more:
            more = False
            pd[idx] = []
            pd[idx + 1] = []

            # extend palindromes with odd lengths
            for x in pd[idx - 2]:
                if (x[0] == 0) or (x[-1] == len(s) - 1):
                    continue
                a = x[0] - 1
                b = x[-1] + 1
                if s[a] == s[b]:
                    pd[idx].append(range(a, b + 1))
                    more = True

            # extend palindromes with even lengths
            for x in pd[idx - 1]:
                if (x[0] == 0) or (x[-1] == len(s) - 1):
                    continue
                a = x[0] - 1
                b = x[-1] + 1
                if s[a] == s[b]:
                    pd[idx].append(range(a, b + 1))
                    more = True

            idx += 2

        return sum([len(pd[x]) for x in pd.keys()])

sol = Solution()
sol.countSubstrings('aaa')