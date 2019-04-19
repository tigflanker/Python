# -*- coding: utf-8 -*-
# Author: tigflanker
# https://leetcode-cn.com/problems/two-sum/

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i, x in enumerate(nums):
            y = target - x
            try:
                if (y in nums) & (i != nums.index(y)):
                    return [i, nums.index(y)]
            except:
                pass 