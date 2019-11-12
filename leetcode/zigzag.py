
from ipdb import set_trace

class Solution:
    def convert(self, s: str, numRows: int) -> str:

        # initialize an object to store data by rows
        rows = dict()
        for x in range(numRows):
            rows[x] = ''

        # create indices
        idx = list(range(numRows)) + list(range(1, numRows - 1))[::-1]
        idx = idx * (int(len(s) / len(idx)) + 1)

        # write to data object
        for x, sx in zip(idx, s):
            rows[x] += sx

        set_trace()

        # return result
        return ''.join([x for x in rows.values()])

sol = Solution()
sol.convert("PAYPALISHIRING", 4)
