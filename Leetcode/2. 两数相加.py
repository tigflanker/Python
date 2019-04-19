# -*- coding: utf-8 -*-
# Author: tigflanker
# https://leetcode-cn.com/problems/add-two-numbers/

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1: List, l2: List) -> List:
        import re

        l1.reverse()
        l2.reverse()

        l3 = str(int(re.sub(r'\D', '', str(l1))) + int(re.sub(r'\D', '', str(l2))))
        l3 = list(map(int, l3))

        l3.reverse()
        return l3