# -*- coding: utf-8 -*-
# Author: tigflanker
# https://leetcode-cn.com/problems/longest-palindromic-substring/submissions/

class Solution:
    import re
    def longestPalindrome(self, s: str) -> str:     
        
        l = list(s)
        l.reverse()
        n = s + ',' + ''.join(l)
        l = len(s)
        
        if l < 2:
            return s
        else:
            for i in range(l,0,-1):
                for j in range(l-i,-1,-1):
                    x = re.search(r'(\w{'+str(j)+r'})(\w{'+str(i)+r'})(\w*),\w{'+str(l-i-j)+r'}\2\w{'+str(j)+r'}', n)
                    if x:
                        return x.group(2)