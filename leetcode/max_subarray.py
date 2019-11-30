
from ipdb import set_trace
from typing import List


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return None
        if len(nums) == 1:
            return nums[0]
        current_max = global_max = nums[0]
        for i in range(1, len(nums)):
            set_trace()
            current_max = max(nums[i], nums[i] + current_max)
            if global_max < current_max:
                global_max = current_max
        return global_max

sol = Solution()
sol.maxSubArray([-2,1,-3,4,-1,2,1,-5,4])