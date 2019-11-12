
from ipdb import set_trace

class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        set_trace()
        val1 = 0
        for digit in num1:
            val1 *= 10
            for d in '0123456789':
                val1 += digit > d
        val2 = 0
        for digit in num2:
            val2 *= 10
            for d in '0123456789':
                val2 += digit > d

        return(str(val1*val2))

sol = Solution()
assert sol.multiply('123', '456') == '56088'
