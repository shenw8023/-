# 300. 最长递增子序列
def solution(nums):
    if not nums:
        return 0
    n = len(nums)
    dp = [1 for i in range(n)]
    for i in range(n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j]+1)
    return max(dp) ##[ ]注意输出的是最大的那个，不是最后一个



# 剑指 Offer 42. 连续子数组的最大和
def maxSubArray(nums) -> int:
    dp = nums
    for i in range(1,len(nums)): #[ ]注意从第二项开始遍历，否则下面i-1会越界
        dp[i] = max(dp[i-1] + nums[i], nums[i])
    return max(dp) 


# 1143. 最长公共子序列
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m = len(text1)
        n = len(text2)
        dp = [[0 for i in range(n+1)] for j in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    

# 72 编辑距离
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0 for i in range(n+1)] for j in range(m+1)]

        for i in range(1, n+1):
            dp[0][i] = i
        for i in range(1, m+1):
            dp[i][0] = i


        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1]== word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])+1

        return dp[m][n] 
    
    
    
# 516. 最长回文子序列   
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        dp = [[0 for j in range(len(s))] for i in range(len(s))]
        for i in range(len(s)):
            for j in range(len(s)):
                if i==j:
                    dp[i][j] = 1
        for i in range(len(s)-2, -1, -1):  # 由于每个位置是由左下的值推出来的，所以遍历方向：从下往上，从左往右
            for j in range(i+1, len(s)):
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        return dp[0][len(s)-1] 
    
    
# 1312. 让字符串成为回文串的最少插入次数 
class Solution:
    def minInsertions(self, s: str) -> int:
        n = len(s)
        dp = [[0 for i in range(n)] for j in range(n)]
        for i in range(n-2, -1, -1):
            for j in range(i+1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1]
                else:
                    dp[i][j] = min(dp[i+1][j], dp[i][j-1]) + 1
        return dp[0][n-1]