# Code Test



```python
Leetcode index:
  21

Question:
  将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

Answer:
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if not list1:
            return list2
        if not list2:
            return list1

        result = ListNode(0)

        p1 = list1
        p2 = list2
        pr = result

        while p1 and p2:
            if p1.val < p2.val:
                pr.next = p1
                p1 = p1.next
            else:
                pr.next = p2
                p2 = p2.next
            pr = pr.next

        if p1:
            pr.next = p1
        if p2:
            pr.next = p2

        return result.next
```



```python
Leetcode index:
  86

Question:
  给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。你应当 保留 两个分区中每个节点的初始相对位置。

Answer:
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        if not head:
            return head

        big = ListNode(0)
        small = ListNode(0)

        pb = big
        ps = small

        while head:
            if head.val < x:
                ps.next = head
                ps = ps.next
            else:
                pb.next = head
                pb = pb.next
            head = head.next

        pb.next = None
        ps.next = big.next
        return small.next
```



```python
Leetcode index:
  23

Question:
  给你一个链表数组，每个链表都已经按升序排列。请你将所有链表合并到一个升序链表中，返回合并后的链表。

Answer:
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # 搞一个优先队列（最小二叉堆）
        import heapq
        prior_queue = []
        # 把 lists 中的头结点都放进去
        i = 0
        for node in lists:
            if node:
                heapq.heappush(prior_queue, (node.val, i, node))
                i += 1
        # 依次 pop 队头获取结果
        result = ListNode(0)
        pr = result
        while prior_queue:
            _, i, node = heapq.heappop(prior_queue)
            pr.next = node
            if node.next:
                heapq.heappush(prior_queue, (node.next.val, i, node.next))
            pr = pr.next
        return result.next
```



```python
Leetcode index:
  19

Question:
  给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

Answer:
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        if not head or not head.next:
            return None

        p_fast = head
        p_slow = head

        p_fast_step = 0
        while head and p_fast_step <= n:
            p_fast = p_fast.next
            p_fast_step += 1

        while p_fast:
            p_fast = p_fast.next
            p_slow = p_slow.next

        p_slow.next = p_slow.next.next
        return head
```



```python
Leetcode index:
  876

Question:
  给你单链表的头结点 head ，请你找出并返回链表的中间结点。如果有两个中间结点，则返回第二个中间结点。

Answer:
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        p_fast = head
        p_slow = head

        while p_fast and p_fast.next:
            p_fast = p_fast.next.next
            p_slow = p_slow.next

        return p_slow
```



```python
Leetcode index:
  142

Question:
  给定一个链表的头节点 head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

Answer:
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        fast, slow = head, head
        # 先判断有没有
        while True:
            if not(fast and fast.next): return
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                break
        # 再找到
        slow = head
        while fast != slow:
            fast = fast.next
            slow = slow.next
        return slow
```



```python
Leetcode index:
  160

Question:
  给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。

Answer:
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        pa = headA
        pb = headB

        while pa != pb:
            if pa:
                pa = pa.next
            else:
                pa = headB
            if pb:
                pb = pb.next
            else:
                pb = headA

        return pa
```

```python
Leetcode index:
  26

Question:
  给你一个 升序排列 的数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。
  元素的 相对顺序 应该保持 一致 。
  由于在某些语言中不能改变数组的长度，所以必须将结果放在数组nums的第一部分。更规范地说，如果在删除重复项之后有 k 个元素，那么 nums 的前 k 个元素应该保存最终结果。将最终结果插入 nums 的前 k 个位置后返回 k 。
  不要使用额外的空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

Answer:
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        slow = 0
        fast = 0

        while fast < len(nums):
            if nums[fast] != nums[slow]:
                slow += 1
                nums[slow] = nums[fast]
            fast += 1
        return slow + 1
```



```python
Leetcode index:
  27
  
Question:
  给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
  
Answer:
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        slow = 0
        fast = 0

        while fast < len(nums):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        return slow
```



```python
Leetcode index:
  167

Question:
  给你一个下标从 1 开始的整数数组 numbers ，该数组已按 非递减顺序排列 ，请你从数组中找出满足相加之和等于目标数 target 的两个数
  如果设这两个数分别是 numbers[index1] 和 numbers[index2] ，则 1 <= index1 < index2 <= numbers.length 。
  以长度为 2 的整数数组 [index1, index2] 的形式返回这两个整数的下标 index1 和 index2。你可以假设每个输入 只对应唯一的答案 ，而且你 不可以 重复使用相同的元素。
  你所设计的解决方案必须只使用常量级的额外空间。

Answer:
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers) - 1

        while left < right:
            tmp_sum = numbers[left] + numbers[right]
            if tmp_sum == target:
                return [left + 1, right + 1]
            elif tmp_sum > target:
                right -= 1
            else:
                left += 1
        return [left + 1, right + 1]
```



```python
Leetcode index:
  344
  
Question:
  编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 s 的形式给出。
  不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。

Answer:
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left = 0
        right = len(s) - 1
        
        while left < right:
            tmp = s[left]
            s[left] = s[right]
            s[right] = tmp
            left += 1
            right -= 1
```



```python
Leetcode index:
  5
  
Question:
  给你一个字符串 s，找到 s 中最长的回文子串。如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。
  
Answer:
class Solution:
    def longestPalindrome(self, s: str) -> str:
        res = ""
        for i in range(len(s)):
            s1 = self.tmpFunction(s, i, i)
            s2 = self.tmpFunction(s, i, i + 1)
            if len(s1) > len(res):
                res = s1
            if len(s2) > len(res):
                res = s2
        return res

    # 以 i, j 为中心的最长回文子串
    def tmpFunction(self, s, left, right) -> str:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1: right]
```



```python
Leetcode index:
  283

Question:
  给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。请注意，必须在不复制数组的情况下原地对数组进行操作。

Answer:
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        slow = 0
        fast = 0

        while fast < len(nums):
            if nums[fast] != 0:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1

        for i in range(slow, len(nums)):
            nums[i] = 0
```



```python
Leetcode index:
  83

Question:
  给定一个已排序的链表的头 head，删除所有重复的元素，使每个元素只出现一次。返回已排序的链表 。

Answer:
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        
        p_slow = head
        p_fast = head

        while p_fast:
            if p_fast.val != p_slow.val:
                p_slow.next = p_fast
                p_slow = p_slow.next
            p_fast = p_fast.next

        p_slow.next = None
        return head
```



```python
Leetcode index:
  104

Question:
  给定一个二叉树，找出其最大深度。二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。说明: 叶子节点是指没有子节点的节点。

Answer:
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def __init__(self):
        self.depth = 0
        self.result = 0
    
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        self.depth += 1
        self.result = max(self.result, self.depth)
        self.maxDepth(root.left)
        self.maxDepth(root.right)
        self.depth -= 1
        return self.result
```



```python
Leetcode index:
  144

Question:
  给你二叉树的根节点 root ，返回它节点值的 前序 遍历。

Answer:
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def __init__(self):
        self.result = []
    
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        self.result.append(root.val)
        self.preorderTraversal(root.left)
        self.preorderTraversal(root.right)
        return self.result
```



```python
Leetcode index:
  543

Question:
  给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。

Answer:
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def __init__(self):
        self.result = 0

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.tmp_function(root)
        return self.result

    # 当前节点作为根节点的最大深度
    def tmp_function(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        else:
            left_depth = self.tmp_function(root.left)
            right_depth = self.tmp_function(root.right)
            tmp = left_depth + right_depth
            self.result = max(tmp, self.result)
            return max(left_depth, right_depth) + 1
```



```python
Leetcode index:
  322

Question:
  给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。你可以认为每种硬币的数量是无限的。

Answer:
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        memory = [-100] * (amount + 1)
        # 定义一个 dp 函数，输入是 coin_list 和 target，输出是凑出 target 的最小硬币数量
        def dp(coin_list, target):
            # 处理基本边界情况
            if target < 0:
                return -1
            if target == 0:
                return 0

            # 读的优化
            if memory[target] != -100:
                return memory[target]

            # 处理常见情况
            result = float("inf")
            for coin in coin_list:
                sub_problem = dp(coin_list, target - coin)
                # 子问题无解则跳过
                if sub_problem == -1:
                    continue
                result = min(result, sub_problem + 1)
            memory[target] = -1 if result == float("inf") else result
            return memory[target]
        return dp(coins, amount)
```



```python
Leetcode index:
  509

Question:
  斐波那契数 （通常用 F(n) 表示）形成的序列称为 斐波那契数列 。该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：F(0) = 0，F(1) = 1 F(n) = F(n - 1) + F(n - 2)，其中 n > 1 给定 n ，请计算 F(n) 。

Answer:
class Solution:
    def fib(self, n: int) -> int:
        # 备忘录全初始化为 0
        memo = [0] * (n + 1)
        # 进行带备忘录的递归
        return self.dp(memo, n)

    # 带着备忘录进行递归
    def dp(self, memo: List[int], n: int) -> int:
        # base case
        if n == 0 or n == 1:
            return n
        # 已经计算过，不用再计算了
        if memo[n] != 0:
            return memo[n]
        memo[n] = self.dp(memo, n - 1) + self.dp(memo, n - 2)
        return memo[n]
```



```python
Leetcode index:
  46

Question:
  给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。

Answer:
class Solution:
    def __init__(self):
        self.result = []
        self.mid_result = []

    def midFunction(self, nums: List[int], length: int):
        if len(self.mid_result) == length:
            self.result.append(self.mid_result.copy())
            return
        else:
            for i in range(len(nums)):
                if nums[i] in self.mid_result:
                    continue
                self.mid_result.append(nums[i])
                self.midFunction(nums, length)
                self.mid_result.pop()

    def permute(self, nums: List[int]) -> List[List[int]]:
        self.midFunction(nums, len(nums))
        return self.result
```



```python
Leetcode index:
  51

Question:
  按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。
  n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
  给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。

Answer:
class Solution:
    def __init__(self):
        self.result = []

    def solveNQueens(self, n: int) -> List[List[str]]:
        board = [["." for _ in range(n)] for _ in range(n)]
        self.travels(board, 0)
        return self.result

    def travels(self, board: List[List[str]], row: int):
        # 找到一种可行解，从上往下放
        if row == len(board):
            self.result.append(["".join(row) for row in board])
            return
        else:
            for col in range(len(board)):
                if not self.valid(board, row, col):
                    continue
                else:
                    board[row][col] = "Q"
                    self.travels(board, row + 1)
                    board[row][col] = "."

    # 因为从上往下放置，所以不需要考虑下方的
    def valid(self, board: List[str], row: int, col: int) -> bool:
        # 验证上方
        for i in range(row):
            current_element = board[i][col]
            if current_element == "Q":
                return False
        # 验证左上方
        current_element_col = col - 1
        current_element_row = row - 1
        while current_element_col >= 0 and current_element_row >= 0:
            current_element = board[current_element_row][current_element_col]
            if current_element == "Q":
                return False
            current_element_col -= 1
            current_element_row -= 1
        # 验证右上方
        current_element_col = col + 1
        current_element_row = row - 1
        while current_element_col < len(board) and current_element_row >= 0:
            current_element = board[current_element_row][current_element_col]
            if current_element == "Q":
                return False
            current_element_col += 1
            current_element_row -= 1
        return True
```



```python
Leetcode index:
  39

Question:
  给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。
  你可以按 任意顺序 返回这些组合。candidates 中的 同一个 数字可以 无限制重复被选取。
  如果至少一个数字的被选数量不同，则两种组合是不同的。对于给定的输入，保证和为 target 的不同组合数少于 150 个。

Answer:
class Solution:
    def __init__(self):
        self.result = []
        self.mid_result = []
        self.travel_sum = 0

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        if len(candidates) == 0:
            return candidates
        self.travels(candidates, target, 0)
        return self.result

    def travels(self, candidates: List[int], target: int, start: int):
        if self.travel_sum == target:
            self.result.append(self.mid_result.copy())
            return

        if self.travel_sum > target:
            return

        for i in range(start, len(candidates)):
            # 裁剪条件
            self.mid_result.append(candidates[i])
            self.travel_sum += candidates[i]
            self.travels(candidates, target, i)
            self.mid_result.pop()
            self.travel_sum -= candidates[i]
```



```python
Leetcode index:
  40

Question:
  给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。candidates 中的每个数字在每个组合中只能使用 一次 。注意：解集不能包含重复的组合。

Answer:
class Solution:
    def __init__(self):
        self.result = []
        self.mid_result = []
        self.track_sum = 0
    
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        self.travels(candidates, target, 0)
        return self.result
    
    def travels(self, nums: List[int], target: int, start: int):
        if self.track_sum == target:
            self.result.append(self.mid_result.copy())
            return 
        
        if self.track_sum > target:
            return 
        
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            self.mid_result.append(nums[i])
            self.track_sum += nums[i]
            self.travels(nums, target, i + 1)
            self.mid_result.pop()
            self.track_sum -= nums[i]
```



```python
Leetcode index:
  78

Question:
  给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。

Answer:
class Solution:
    def __init__(self):
        self.result = []
        self.mid_result = []

    def subsets(self, nums: List[int]) -> List[List[int]]:
        self.travels(nums, 0)
        return self.result

    def travels(self, nums: List[int], start: int):
        self.result.append(self.mid_result.copy())
        if start == len(nums):
            return

        for i in range(start, len(nums)):
            self.mid_result.append(nums[i])
            self.travels(nums, i + 1)
            self.mid_result.pop()
```



```python
Leetcode index:
  90

Question:
  给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。

Answer:
class Solution:
    def __init__(self):
        self.result = []
        self.mid_result = []

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        self.travels(nums, 0)
        return self.result

    def travels(self, nums: List[int], start: int):
        self.result.append(self.mid_result.copy())

        if start == len(nums):
            return

        for i in range(start, len(nums)):
            # 如果是同一层，那么只要左边的
            if i > start and nums[i] == nums[i - 1]:
                continue
            self.mid_result.append(nums[i])
            self.travels(nums, i + 1)
            self.mid_result.pop()

```



```python
Leetcode index:
  111

Question:
  给定一个二叉树，找出其最小深度。最小深度是从根节点到最近叶子节点的最短路径上的节点数量。说明：叶子节点是指没有子节点的节点。

Answer:
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from collections import deque
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        q = deque()
        min_depth = 1
        q.append(root)

        while len(q) > 0:
            size = len(q)
            for i in range(size):
                cur = q.popleft()
                if not cur.left and not cur.right:
                    return min_depth
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
            min_depth += 1
        return min_depth
```



```python
Leetcode index:
  752

Question:
  你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 。每个拨轮可以自由旋转：例如把 '9' 变为 '0'，'0' 变为 '9' 。
  每次旋转都只能旋转一个拨轮的一位数字。锁的初始数字为 '0000' ，一个代表四个拨轮的数字的字符串。
  列表 deadends 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。
  字符串 target 代表可以解锁的数字，你需要给出解锁需要的最小旋转次数，如果无论如何不能解锁，返回 -1 。

Answer:
from collections import deque
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        dead_set = set(deadends)
        visited = set()
        q = deque()
        q.append("0000")
        visited.add("0000")

        step = 0

        while len(q) > 0:
            size = len(q)
            for _ in range(size):
                cur = q.popleft()

                if cur in dead_set:
                    continue

                if cur == target:
                    return step

                for i in range(4):
                    plus_one_result = self.plus_one(cur, i)
                    minus_one_result = self.minus_one(cur, i)
                    if plus_one_result not in visited:
                        visited.add(plus_one_result)
                        q.append(plus_one_result)
                    if minus_one_result not in visited:
                        visited.add(minus_one_result)
                        q.append(minus_one_result)
            step += 1

        return -1

    @staticmethod
    def plus_one(s: str, i: int) -> str:
        s_list = list(s)
        s_num_list = [int(e) for e in s_list]
        if s_num_list[i] == 9:
            s_num_list[i] = 0
        else:
            s_num_list[i] += 1
        return "".join([str(e) for e in s_num_list])

    @staticmethod
    def minus_one(s: str, i: int) -> str:
        s_list = list(s)
        s_num_list = [int(e) for e in s_list]
        if s_num_list[i] == 0:
            s_num_list[i] = 9
        else:
            s_num_list[i] -= 1
        return "".join([str(e) for e in s_num_list])
```



```python
Leetcode index:
  Dummy

Question:
  二分查找代码框架

Answer:

def binary_search(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        elif nums[mid] == target:
            # 直接返回
            return mid
    # 直接返回
    return -1

def left_bound(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        elif nums[mid] == target:
            # 别返回，锁定左侧边界
            right = mid - 1
    # 判断 target 是否存在于 nums 中
    # 此时 target 比所有数都大，返回 -1
    if left == len(nums):
        return -1
    # 判断一下 nums[left] 是不是 target
    return left if nums[left] == target else -1

def right_bound(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        elif nums[mid] == target:
            # 别返回，锁定右侧边界
            left = mid + 1
    # 此时 left - 1 索引越界
    if left - 1 < 0:
        return -1
    # 判断一下 nums[left] 是不是 target
    return left - 1 if nums[left - 1] == target else -1
```



```python
Leetcode index:
  3

Question:
  给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。

Answer:
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 先初始化一些变量
        # 搜索边界为 right == len(s)
        # 收缩条件为 window 中出现了重复
        left = 0
        right = 0
        window = {}
        result = 0

        while right < len(s):
            cur_in_char = s[right]
            right += 1

            # 对窗口进行操作
            window = self.dict_add(window, cur_in_char)

            while window[cur_in_char] > 1:
                cur_out_char = s[left]
                left += 1
                window[cur_out_char] -= 1
                
            result = max(right - left, result)
        return result

    @staticmethod
    def dict_add(source_dict, key):
        if key not in source_dict:
            source_dict[key] = 1
        else:
            source_dict[key] += 1
        return source_dict
```



```python
Leetcode index:
  438

Question:
  给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。

Answer:
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        # 先初始化一些基本变量
        # 因为是异位词，所以不用考虑位置
        # 搜索空间为 right < len(s)
        # 收缩条件为 right - left == len(p)
        left = 0
        right = 0
        valid = 0
        window = {}
        need = {}
        result = []
        for c in p:
            need = self.dict_add(need, c)

        while right < len(s):
            cur_in_char = s[right]
            right += 1

            # 窗口操作
            if cur_in_char in need:
                window = self.dict_add(window, cur_in_char)
                if window[cur_in_char] == need[cur_in_char]:
                    valid += 1

            # 收缩窗口
            while (right - left) == len(p):
                if valid == len(need):
                    result.append(left)

                out_cur_char = s[left]
                left += 1
                if out_cur_char in need:
                    if need[out_cur_char] == window[out_cur_char]:
                        valid -= 1
                    window[out_cur_char] -= 1
        return result

    @staticmethod
    def dict_add(source_dict, key):
        if key not in source_dict:
            source_dict[key] = 1
        else:
            source_dict[key] += 1
        return source_dict
```



```python
Leetcode index:
  567

Question:
  给你两个字符串 s1 和 s2 ，写一个函数来判断 s2 是否包含 s1 的排列。如果是，返回 true ；否则，返回 false 。换句话说，s1 的排列之一是 s2 的 子串 。

Answer:
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        # 先初始化一些值
        left = 0
        right = 0
        valid = 0
        window = {}
        need = {}
        # 记录下边界条件
        for c in s1:
            need = self.dict_add(need, c)

        # 先找到可行解
        while right < len(s2):
            in_cur_char = s2[right]
            right += 1

            # 窗口操作
            if in_cur_char in need:
                window = self.dict_add(window, in_cur_char)
                if window[in_cur_char] == need[in_cur_char]:
                    valid += 1

            # 窗口压缩
            while (right - left) == len(s1):
                if len(need) == valid:
                    return True
                out_cur_char = s2[left]
                left += 1
                # 窗口操作
                if out_cur_char in need:
                    if window[out_cur_char] == need[out_cur_char]:
                        valid -= 1
                    window[out_cur_char] -= 1
        return False

    @staticmethod
    def dict_add(source_dict, key):
        if key not in source_dict:
            source_dict[key] = 1
        else:
            source_dict[key] += 1
        return source_dict
```



```python
Leetcode index:
  76

Question:
  给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

Answer:
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # 初始化一些对象
        left = 0
        right = 0
        start = 0
        length = float("inf")
        valid = 0
        window = {}
        need_dict = {}
        # 先统计需要的元素及其个数
        for c in t:
            need_dict = self.dict_add(need_dict, c)

        while right < len(s):
            in_cur_char = s[right]
            right += 1

            # 对两个窗口操作
            if in_cur_char in need_dict:
                window = self.dict_add(window, in_cur_char)
                if window[in_cur_char] == need_dict[in_cur_char]:
                    valid += 1
            # 缩小窗口
            while valid == len(need_dict):
                # 记录结果
                if (right - left) < length:
                    start = left
                    length = right - left

                # 缩小窗口
                out_cur_char = s[left]
                left += 1
                if out_cur_char in window:
                    window[out_cur_char] -= 1
                    if window[out_cur_char] < need_dict[out_cur_char]:
                        valid -= 1
        return "" if length == float("inf") else s[start: start + length]

    @staticmethod
    def dict_add(source_dict, key):
        if key not in source_dict:
            source_dict[key] = 1
        else:
            source_dict[key] += 1
        return source_dict
```



```python
Leetcode index:
  121

Question:
  给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

Answer:
class Solution:
    """
        定义三种状态 iks，定义 dp 为最大利润，定义买入为一次操作
        转移方程
            dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
            dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
        base case
        for i in range(len(prices)):
            dp[i][0][0] = 0
            dp[i][0][1] = -float("inf")
        for k in range(max_k, 0, -1):
            dp[0][k][0] = 0
            dp[0][k][1] = -prices[0]

        下面这道题 k == 1
        转移方程
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            dp[i][1] = max(dp[i-1][1], - prices[i])
        base case
        for i in range(len(prices)):
            dp[i][0] = 0
            dp[i][1] = -float("inf")
        for k in range(max_k, 0, -1):
            dp[0][0] = 0
            dp[0][1] = -prices[0]
    """
    def maxProfit(self, prices: List[int]) -> int:
        # 先判断极端情况
        if len(prices) <= 0:
            return 0
        # 初始化一些变量
        dp = [[0, 0.0] for _ in range(len(prices))]
        # base case
        for i in range(len(prices)):
            dp[i][0] = 0
            dp[i][1] = -float("inf")
        # 寻找最优解
        for i in range(len(prices)):
            if i == 0:
                dp[i][0] = 0
                dp[i][1] = -prices[0]
                continue
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            dp[i][1] = max(dp[i-1][1], -prices[i])
        return dp[len(prices) - 1][0]
```



```python
Leetcode index:
  122

Question:
  给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。返回 你能获得的 最大 利润 。

Answer:
class Solution:
    """
        定义三种状态 iks，定义 dp[i][k][s] 为最大收益，定义买入为交易操作
        状态转移方程
            dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + price[i])
            dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - price[i])
        base case
        for i in range(len(prices))):
            dp[i][0][0] = 0
            dp[i][0][1] = -float("inf")
        for k in range(max_k, 0, -1):
            dp[0][k][0] = 0
            dp[0][k][1] = -prices[0]

        这道题目 k 等于无数次
        状态转移方程
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + price[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - price[i])
        base case
        for i in range(len(prices))):
            dp[i][0] = 0
            dp[i][1] = -float("inf")
    """
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) <= 0:
            return 0
        dp = [[0, 0.0] for _ in range(len(prices))]
        for i in range(len(prices)):
            dp[i][0] = 0
            dp[i][1] = -float("inf")
        for i in range(len(prices)):
            if i == 0:
                dp[i][0] = 0
                dp[i][1] = -prices[i]
                continue
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
        return dp[len(prices)-1][0]
```



```python
Leetcode index:
  123

Question:
  给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

Answer:
class Solution:
    """
        dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
        dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
        for i in range(len(prices)):
            dp[i][0][0] = 0
            dp[i][0][1] = -float("inf")

        k == 2
        dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
        dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
        for i in range(len(prices)):
            dp[i][0] = 0
            dp[i][1] = -float("inf")
    """
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) <= 0:
            return 0
        dp = [[[0, 0.0] for _ in range(3)] for _ in range(len(prices))]
        for i in range(len(prices)):
            dp[i][0][0] = 0
            dp[i][0][1] = -float("inf")
        for i in range(len(prices)):
            for k in range(2, 0, -1):
                if i == 0:
                    dp[i][k][0] = 0
                    dp[i][k][1] = -prices[i]
                    continue
                dp[i][k][0] = max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i])
                dp[i][k][1] = max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i])
        return dp[len(prices) - 1][2][0]
```



```python
Leetcode index:
  188

Question:
  给定一个整数数组 prices ，它的第 i 个元素 prices[i] 是一支给定的股票在第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

Answer:
class Solution:
    """
        基本框架，变量有 i，k，s 三个，s 表示当前持有状态，买入为一次交易
        定义 dp[i][k][s] 为 在 i，k，s 的情况下的最大收益
        状态转移方程为
        dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
        dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
        base case 为
        if i < 0
            dp[-1][k][0] = 0
            dp[-1][k][1] = -float("inf")
        if k == 0
            dp[-1][0][0] = 0
            dp[-1][0][1] = -float("inf")
    """
    def maxProfit(self, k: int, prices: List[int]) -> int:
        # 极端情况
        if len(prices) <= 0:
            return 0
        # 初始化 dp，定义 dp 为 iks 下最大的利润
        max_k = k
        dp = [[[0, 0.0] for _ in range(max_k + 1)] for _ in range(len(prices))]
        # base case
        for i in range(len(prices)):
            dp[i][0][0] = 0
            dp[i][0][1] = -float("inf")
        # 寻找最优解
        for i in range(len(prices)):
            for k in range(max_k, 0, -1):
                if i == 0:
                    dp[i][k][0] = 0
                    dp[i][k][1] = -prices[i]
                    continue
                dp[i][k][0] = max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i])
                dp[i][k][1] = max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i])
        return int(dp[len(prices) - 1][max_k][0])
```



```python
Leetcode index:
  309

Question:
  给定一个整数数组 prices，其中第 prices[i] 表示第 i 天的股票价格 。​设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

Answer:
class Solution:
    """
        dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
        dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
        for i in range(len(prices)):
            dp[i][0][0] = 0
            dp[i][0][1] = -float("inf")

        冷冻限制买入，k 为无限
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
        dp[i][1] = max(dp[i-1][1], dp[i-2][0] - prices[i])
        for i in range(len(prices)):
            dp[i][0] = 0
            dp[i][1] = -float("inf")
    """
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) <= 0:
            return 0
        dp = [[0, 0.0] for _ in range(len(prices))]
        for i in range(len(prices)):
            dp[i][0] = 0
            dp[i][1] = -float("inf")
        for i in range(len(prices)):
            if i == 0:
                dp[i][0] = 0
                dp[i][1] = -prices[i]
                continue
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 2][0] - prices[i])
        return dp[len(prices) - 1][0]
```



```python
Leetcode index:
  714

Question:
  给定一个整数数组 prices，其中 prices[i]表示第 i 天的股票价格 ；整数 fee 代表了交易股票的手续费用。你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。返回获得利润的最大值。注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。

Answer:
class Solution:
    """
        dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
        dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
        for i in range(len(prices)):
            dp[i][0][0] = 0
            dp[i][0][1] = -float("inf")

        k 为无限，手续费在买入时付出
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
        dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i] - fee)
        for i in range(len(prices)):
            dp[i][0] = 0
            dp[i][1] = -float("inf")
    """
    def maxProfit(self, prices: List[int], fee: int) -> int:
        if len(prices) <= 0:
            return 0
        dp = [[0, 0.0] for _ in range(len(prices))]
        for i in range(len(prices)):
            dp[i][0] = 0
            dp[i][1] = -float("inf")
        for i in range(len(prices)):
            if i == 0:
                dp[i][0] = 0
                dp[i][1] = -prices[i] - fee
                continue
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i] - fee)
        return dp[len(prices) - 1][0]
```



```python
Leetcode index:
  198

Question:
  你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

Answer:
class Solution:
    def rob(self, nums: List[int]) -> int:
        """
            定义状态为到达第 i 个房间，选择为拿或者不拿，dp[i] 为当前最大金额
            转移方程为
                dp[i] = max(dp[i-2] + nums[i], dp[i-1])
            base case
            只有一间
                dp[0] = nums[0]
            只有两间
                dp[1] = max(nums[0], nums[1])
        """
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        dp = [0] * len(nums)
        # base case
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums), 1):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
        return dp[len(nums) - 1]
```



```python
Leetcode index:
  213

Question:
  你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。

Answer:
class Solution:
    def rob(self, nums: List[int]) -> int:
        """
        dp[i] = max(dp[i-2] + nums[i], dp[i-1])
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        在上面的基础上加上了约束是首尾相连，因此
            n = len(nums) - 1
            dp[n] = max(
                dp[n-2] + nums[-1] - nums[0],
                dp[n-1]
            )
        """
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        n = len(nums)
        need_first_result = self.mid_function(nums[0: n - 1].copy())
        need_last_result = self.mid_function(nums[1: n].copy())
        print(need_first_result)
        print(need_last_result)
        return max(need_first_result, need_last_result)

    @staticmethod
    def mid_function(nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        dp = [0] * len(nums)
        # base case
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        n = len(nums)
        for i in range(2, n, 1):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
        return dp[n - 1]
```



```python
Leetcode index:
  337

Question:
  小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 root 。除了 root 之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫 ，房屋将自动报警。给定二叉树的 root 。返回 在不触动警报的情况下 ，小偷能够盗取的最高金额 。

Answer:
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def __init__(self):
        self.result = 0

    def rob(self, root: Optional[TreeNode]) -> int:
        left, right = self.travels(root)
        return max(left, right)

    # 从当前节点出发，可以获得的左侧最大值和右侧最大值
    def travels(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return [0, 0]
        left = self.travels(root.left)
        right = self.travels(root.right)
        return [
            max(left[0], left[1]) + max(right[0], right[1]),
            left[0] + right[0] + root.val
        ]
```



```python
Leetcode index:
  1

Question:
  给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target 的那 两个 整数，并返回它们的数组下标。你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。你可以按任意顺序返回答案。

Answer:
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # hash map
        result_dict = {}
        for i, num in enumerate(nums):
            result_dict[num] = i

        for i, num in enumerate(nums):
            other = target - num
            if other not in result_dict:
                continue
            elif i == result_dict[other]:
                continue
            else:
                return [i, result_dict[other]]
```



```python
Leetcode index:
  206

Question:
  给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。

Answer:
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    # def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    #     if not head or not head.next:
    #         return head
    #     last = self.reverseList(head.next)
    #     head.next.next = head
    #     head.next = None
    #     return last

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head

        pre = None
        cur = head
        while cur:
            post = cur.next
            cur.next = pre
            pre = cur
            cur = post
        return pre
```



```python
Leetcode index:
  92

Question:
  给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表。

Answer:
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def __init__(self):
        self.post = ListNode(0)

    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        if left == 1:
            return self.reverseN(head, right)
        head.next = self.reverseBetween(head.next, left - 1, right - 1)
        return head

    def reverseN(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        if n == 1:
            self.post = head.next
            return head
        last = self.reverseN(head.next, n - 1)
        head.next.next = head
        head.next = self.post
        return last
```



```python
Leetcode index:
  25

Question:
  给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

Answer:
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head:
            return head

        start = head
        end = head

        for i in range(k):
            if not end:
                return head
            end = end.next

        next_head = self.reverseBetween(start, end)
        start.next = self.reverseKGroup(end, k)
        return next_head

    @staticmethod
    def reverseBetween(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
        if not a or not a.next:
            return a

        pre = None
        cur = a

        while cur != b:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return pre
```



```python
Leetcode index:
  234

Question:
  给你一个单链表的头节点 head ，请你判断该链表是否为回文链表。如果是，返回 true ；否则，返回 false 。

Answer:
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
import copy
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        s = copy.deepcopy(head)
        r = self.reverse_1(head)

        while s and r:
            if s.val != r.val:
                return False
            s = s.next
            r = r.next
        return True

    def reverse_1(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        last = self.reverse_1(head.next)
        head.next.next = head
        head.next = None
        return last

    @staticmethod
    def reverse_2(head: Optional[ListNode]) -> Optional[ListNode]:
        pre = None
        cur = head

        while cur:
            post = cur.next
            cur.next = pre
            pre = cur
            cur = post

        return pre
```



```python
Leetcode index:
  303

Question:
  给定一个整数数组 nums，处理以下类型的多个查询:
  计算索引 left 和 right （包含 left 和 right）之间的 nums 元素的 和 ，其中 left <= right
  实现 NumArray 类：
  NumArray(int[] nums) 使用数组 nums 初始化对象
  int sumRange(int i, int j) 返回数组 nums 中索引 left 和 right 之间的元素的 总和 ，包含 left 和 right 两点（也就是 nums[left] + nums[left + 1] + ... + nums[right] )
  
Answer:
class NumArray:
    def __init__(self, nums: List[int]):
        self.nums = nums
        self.pre_sum = self.preSum(nums)

    def sumRange(self, left: int, right: int) -> int:
        result = self.pre_sum[right + 1] - self.pre_sum[left]
        return result

    @staticmethod
    def preSum(nums: List[int]) -> List[int]:
        pre_sum_list = [0] * (len(nums) + 1)
        for i in range(1, len(pre_sum_list)):
            pre_sum_list[i] = pre_sum_list[i - 1] + nums[i - 1]
        return pre_sum_list
```



```python
Leetcode index:
  304

Question:
  给定一个二维矩阵 matrix，以下类型的多个请求：
  计算其子矩形范围内元素的总和，该子矩阵的 左上角 为 (row1, col1) ，右下角 为 (row2, col2) 。
  实现 NumMatrix 类：
  NumMatrix(int[][] matrix) 给定整数矩阵 matrix 进行初始化
  int sumRegion(int row1, int col1, int row2, int col2) 返回 左上角 (row1, col1) 、右下角 (row2, col2) 所描述的子矩阵的元素总和 。

Answer:
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.matrix = matrix
        self.m = len(matrix)
        self.n = len(matrix[0])
        self.matrix_pre_sum = [
            [0 for _ in range(self.n + 1)] for _ in range(self.m + 1)
        ]
        for i in range(1, self.m + 1):
            for j in range(1, self.n + 1):
                self.matrix_pre_sum[i][j] = self.matrix_pre_sum[i - 1][j] \
                                            + self.matrix_pre_sum[i][j - 1] \
                                            + self.matrix[i - 1][j - 1] \
                                            - self.matrix_pre_sum[i - 1][j - 1]

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        result = self.matrix_pre_sum[row2 + 1][col2 + 1] \
                 + self.matrix_pre_sum[row1][col1] \
                 - self.matrix_pre_sum[row2 + 1][col1] \
                 - self.matrix_pre_sum[row1][col2 + 1]
        return result
```



```python
Leetcode index:
  528

Question:
  给你一个 下标从 0 开始 的正整数数组 w ，其中 w[i] 代表第 i 个下标的权重。
  请你实现一个函数 pickIndex ，它可以 随机地 从范围 [0, w.length - 1] 内（含 0 和 w.length - 1）选出并返回一个下标。选取下标 i 的 概率 为 w[i] / sum(w) 。
  例如，对于 w = [1, 3]，挑选下标 0 的概率为 1 / (1 + 3) = 0.25 （即，25%），而选取下标 1 的概率为 3 / (1 + 3) = 0.75（即，75%）。

Answer:
import random
class Solution:
    def __init__(self, w: List[int]):
        self.w_length = len(w)
        self.pre_sum = [0] * (self.w_length + 1)
        for i in range(1, len(self.pre_sum)):
            self.pre_sum[i] = self.pre_sum[i - 1] + w[i - 1]

    def pickIndex(self) -> int:
        length = len(self.pre_sum)
        target = random.randint(1, self.pre_sum[length - 1])
        return self.low_bound(self.pre_sum, target) - 1

    @staticmethod
    def low_bound(nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = int(left + (right - left) / 2)
            if nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            elif nums[mid] == target:
                right = mid - 1
        return left
```



```python
Leetcode index:
  1011

Question:
  传送带上的包裹必须在 days 天内从一个港口运送到另一个港口。
  传送带上的第 i 个包裹的重量为 weights[i]。每一天，我们都会按给出重量（weights）的顺序往传送带上装载包裹。我们装载的重量不会超过船的最大运载重量。
  返回能在 days 天内将传送带上的所有包裹送达的船的最低运载能力。

Answer:
class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        left = max(weights)
        right = sum(weights)

        while left <= right:
            mid = int(left + (right - left) / 2)

            need = 1
            cur = 0
            for w in weights:
                if (cur + w) > mid:
                    need += 1
                    cur = 0
                cur += w

            if need < days:
                right = mid - 1
            elif need > days:
                left = mid + 1
            elif need == days:
                right = mid - 1
        return left
```



```python
Leetcode index:
  410

Question:
  给定一个非负整数数组 nums 和一个整数 m ，你需要将这个数组分成 m 个非空的连续子数组。
  设计一个算法使得这 m 个子数组各自和的最大值最小。

Answer:
class Solution:
    def splitArray(self, nums: List[int], k: int) -> int:
        left = max(nums)
        right = sum(nums)
        while left <= right:
            mid = int(left + (right - left) / 2)
            if self.check(nums, mid, k):
                right = mid - 1
            else:
                left = mid + 1
        return left

    @staticmethod
    def check(nums: List[int], target: int, k: int):
        _sum = 0
        count = 1
        for i in range(len(nums)):
            if (_sum + nums[i]) > target:
                count += 1
                _sum = nums[i]
            else:
                _sum += nums[i]
        return count <= k
```



```python
Leetcode index:
  875

Question:
  珂珂喜欢吃香蕉。这里有 n 堆香蕉，第 i 堆中有 piles[i] 根香蕉。警卫已经离开了，将在 h 小时后回来。
  珂珂可以决定她吃香蕉的速度 k （单位：根/小时）。每个小时，她将会选择一堆香蕉，从中吃掉 k 根。如果这堆香蕉少于 k 根，她将吃掉这堆的所有香蕉，然后这一小时内不会再吃更多的香蕉。
  珂珂喜欢慢慢吃，但仍然想在警卫回来前吃掉所有的香蕉。
  返回她可以在 h 小时内吃掉所有香蕉的最小速度 k（k 为整数）。

Answer:
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        left = 1
        right = sum(piles)

        while left <= right:
            mid = int(left + (right - left) / 2)
            if self.could_eat_all(piles, h, mid):
                right = mid - 1
            else:
                left = mid + 1
        return left

    @staticmethod
    def could_eat_all(piles: List[int], h: int, speed: int):
        cost = 0
        for p in piles:
            if p <= speed:
                cost += 1
            else:
                _cost = p / speed
                int_cost = int(_cost)
                if int_cost == _cost:
                    cost += int_cost
                else:
                    cost += int_cost + 1
        return cost <= h
```



```python
Leetcode index:
  870

Question:
  给定两个大小相等的数组 nums1 和 nums2，nums1 相对于 nums2 的优势可以用满足 nums1[i] > nums2[i] 的索引 i 的数目来描述。
  返回 nums1 的任意排列，使其相对于 nums2 的优势最大化。

Answer:
import heapq
class Solution:
    def advantageCount(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        maxpq = [(-val, i) for i, val in enumerate(nums2)]
        heapq.heapify(maxpq)

        nums1_left = 0
        nums1_right = len(nums1) - 1

        new_nums_1 = [0] * len(nums1)

        while maxpq:
            val, i = heapq.heappop(maxpq)
            num2_element = -val
            if num2_element < nums1[nums1_right]:
                new_nums_1[i] = nums1[nums1_right]
                nums1_right -= 1
            else:
                new_nums_1[i] = nums1[nums1_left]
                nums1_left += 1
        return new_nums_1
```



```python
Leetcode index:
  704

Question:
  给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

Answer:
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = int(left + (right - left) / 2)
            if nums[mid] == target:
                return mid
            if nums[mid] > target:
                right = mid - 1
            if nums[mid] < target:
                left = mid + 1
        return -1
```



```python
Leetcode index:
  114

Question:
  给你二叉树的根结点 root ，请你将它展开为一个单链表：
  展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
  展开后的单链表应该与二叉树 先序遍历 顺序相同。

Answer:
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if not root:
            return None
        
        self.flatten(root.left)
        self.flatten(root.right)

        left = root.left
        right = root.right

        root.left = None
        root.right = left

        p = root
        while p.right:
            p = p.right
        p.right = right
```



```python
Leetcode index:
  116

Question:
  给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：
  struct Node {
  	int val;
  	Node *left;
  	Node *right;
  	Node *next;
	}
	填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
  初始状态下，所有 next 指针都被设置为 NULL。

Answer:
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return root
        self.travels(root.left, root.right)
        return root

    def travels(self, node1: 'Optional[Node]', node2: 'Optional[Node]'):
        if not node1 or not node2:
            return None
        node1.next = node2
        self.travels(node1.left, node1.right)
        self.travels(node2.left, node2.right)
        self.travels(node1.right, node2.left)
```



```python
Leetcode index:
  226

Question:
  给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。

Answer:
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        self.travels(root)
        return root

    def travels(self, root: Optional[TreeNode]):
        if not root:
            return 
        tmp = root.left
        root.left = root.right
        root.right = tmp
        self.travels(root.left)
        self.travels(root.right)
```



```python
Leetcode index:
  105

Question:
  给定两个整数数组 preorder 和 inorder ，其中 preorder 是二叉树的先序遍历， inorder 是同一棵树的中序遍历，请构造二叉树并返回其根节点。

Answer:
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if len(preorder) == 0 or len(inorder) == 0:
            return None

        root = TreeNode(preorder[0])
        index = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1: index + 1], inorder[0: index])
        root.right = self.buildTree(preorder[index + 1:], inorder[index + 1:])
        return root
```



```python
Leetcode index:
  106

Question:
  给定两个整数数组 inorder 和 postorder ，其中 inorder 是二叉树的中序遍历， postorder 是同一棵树的后序遍历，请你构造并返回这颗 二叉树 。

Answer:
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if len(inorder) == 0 or len(postorder) == 0:
            return None
        root = TreeNode(postorder[-1])
        index = inorder.index(postorder[-1])
        root.left = self.buildTree(inorder[:index], postorder[:index])
        root.right = self.buildTree(inorder[index + 1:], postorder[index: -1])
        return root
```



```python
Leetcode index:
  654

Question:
  给定一个不重复的整数数组 nums 。 最大二叉树 可以用下面的算法从 nums 递归地构建:
    创建一个根节点，其值为 nums 中的最大值。
    递归地在最大值 左边 的 子数组前缀上 构建左子树。
    递归地在最大值 右边 的 子数组后缀上 构建右子树。
  返回 nums 构建的 最大二叉树 。

Answer:
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        if len(nums) == 0:
            return None
        # 找到最大值及对应的索引，最大值作为 root 节点
        # 索引左右两边，分别递归构建子树
        max_val = -float("inf")
        max_index = -1
        for i in range(0, len(nums)):
            cur = nums[i]
            if cur > max_val:
                max_val = cur
                max_index = i
        root = TreeNode(max_val)
        root.left = self.constructMaximumBinaryTree(nums[: max_index])
        root.right = self.constructMaximumBinaryTree(nums[max_index + 1:])
        return root
```



```python
Leetcode index:
  230

Question:
  给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 个最小元素（从 1 开始计数）。

Answer:
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def __init__(self):
        self.rank = 0
        self.result = 0

    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.travels(root, k)
        return self.result

    def travels(self, root: Optional[TreeNode], k: int):
        if not root:
            return 0
        self.travels(root.left, k)
        self.rank += 1
        if self.rank == k:
            self.result = root.val
        self.travels(root.right, k)
```



```python
Leetcode index:
  700

Question:
  给定二叉搜索树（BST）的根节点 root 和一个整数值 val。
  你需要在 BST 中找到节点值等于 val 的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 null 。

Answer:
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return None
        root_val = root.val
        if root_val == val:
            return root
        elif root_val > val:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)
```



```python
Leetcode index:
  98

Question:
  给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。

Answer:
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        return self._isValidBST(root, None, None)

    def _isValidBST(self, root: Optional[TreeNode], min_node: Optional[TreeNode], max_node: Optional[TreeNode]) -> bool:
        if not root:
            return True
        if min_node and min_node.val >= root.val:
            return False
        if max_node and max_node.val <= root.val:
            return False
        return self._isValidBST(root.left, min_node, root) and self._isValidBST(root.right, root, max_node)
```



```python
Leetcode index:
  797

Question:
  给你一个有 n 个节点的 有向无环图（DAG），请你找出所有从节点 0 到节点 n-1 的路径并输出（不要求按特定顺序）
  graph[i] 是一个从节点 i 可以访问的所有节点的列表（即从节点 i 到节点 graph[i][j]存在一条有向边）。

Answer:
class Solution:
    def __init__(self):
        self.result = []
        self.mid_result = []

    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        self.travels(graph, 0)
        return self.result

    def travels(self, graph: List[List[int]], depth: int):
        self.mid_result.append(depth)

        graph_length = len(graph)

        if depth == graph_length - 1:
            self.result.append(self.mid_result.copy())

        for node in graph[depth]:
            self.travels(graph, node)

        self.mid_result.pop()
```



```python
Leetcode index:
  130

Question:
  给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。

Answer:
class Solution:
    def __init__(self):
        self.o = "O"
        self.x = "X"
        self.director = [[1, 0], [0, 1], [0, -1], [-1, 0]]

    def solve(self, board: List[List[str]]) -> None:
        if not board:
            return
        # 二维数组拉成一维 [x, y] m, n => x * n + y
        m = len(board)
        n = len(board[0])
        uf = UnionFind(m * n + 1)
        dummy = m * n
        # 列边
        for i in range(m):
            if board[i][0] == self.o:
                uf.union(dummy, n * i + 0)
            if board[i][n - 1] == self.o:
                uf.union(dummy, n * i + n - 1)
        # 行边
        for j in range(n):
            if board[0][j] == self.o:
                uf.union(dummy, 0 * n + j)
            if board[m - 1][j] == self.o:
                uf.union(dummy, n * (m - 1) + j)
        # 联通内部的 self.o
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if board[i][j] == self.o:
                    for d in self.director:
                        x = i + d[0]
                        y = j + d[1]
                        if board[x][y] == self.o:
                            uf.union(x * n + y, i * n + j)
        # 替换
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if not uf.is_connected(dummy, i * n + j):
                    board[i][j] = self.x


class UnionFind:
    def __init__(self, n):
        self.count = n
        self.parent = [0] * n
        for i in range(n):
            self.parent[i] = i

    def find_parent(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find_parent(self.parent[node])
        return self.parent[node]

    def union(self, node_1, node_2):
        root_1 = self.find_parent(node_1)
        root_2 = self.find_parent(node_2)
        if root_1 == root_2:
            return
        self.parent[root_2] = root_1
        self.count -= 1

    def is_connected(self, node_1, node_2):
        return self.find_parent(node_1) == self.find_parent(node_2)

    def get_count(self):
        return self.count
```



```python
Leetcode index:
  146

Question:
  请你设计并实现一个满足 LRU (最近最少使用) 缓存 约束的数据结构。
  实现 LRUCache 类：
  LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
  int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
  void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 
  如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
  函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

Answer:
class DoubleListNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class DoubleList:
    def __init__(self):
        self.head = DoubleListNode(0, 0)
        self.tail = DoubleListNode(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def get_size(self):
        return self.size

    def addNodeLast(self, node: DoubleListNode):
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node
        self.size += 1

    def removeNode(self, node: DoubleListNode):
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1

    def removeFirst(self):
        if self.head.next == self.tail:
            return None
        first = self.head.next
        self.removeNode(first)
        return first


class LRUCache:
    """
        1. 使用 双向链表 + Map 实现
        2. 双向链表需要实现：
            a. 基本属性：尺寸
            b. 需要实现的接口：某一节点的删除；删除第一个节点；增加不存在的节点
        3. LRU 中间接口：
            a. 增加某一节点
            b. 删除某一节点
            c. 删除最久远节点
            d. 提升节点的时效性
        4. LRU 基本属性：
            a. 缓存空间大小
            b. Map
            c. 双向链表作为 Cache
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = DoubleList()
        # int DoubleListNode
        self.kv_map = {}

    def remove_kv(self, key):
        node = self.kv_map[key]
        self.cache.removeNode(node)
        self.kv_map.pop(key)

    def add_kv(self, key, value):
        node = DoubleListNode(key, value)
        self.cache.addNodeLast(node)
        self.kv_map[key] = node

    def remove_early(self):
        node = self.cache.removeFirst()
        self.kv_map.pop(node.key)

    def improve_kv_timeliness(self, key):
        node = self.kv_map[key]
        self.cache.removeNode(node)
        self.cache.addNodeLast(node)

    def get(self, key: int) -> int:
        if key not in self.kv_map:
            return -1
        self.improve_kv_timeliness(key)
        value = self.kv_map[key].value
        return value

    def put(self, key: int, value: int) -> None:
        if key in self.kv_map:
            self.remove_kv(key)
            self.add_kv(key, value)
            return None
        if self.capacity == self.cache.get_size():
            self.cache.removeFirst()
        self.add_kv(key, value)
        return None
```



```python
Leetcode index:
  496

Question:
  nums1 中数字 x 的 下一个更大元素 是指 x 在 nums2 中对应位置 右侧 的 第一个 比 x 大的元素。
  给你两个 没有重复元素 的数组 nums1 和 nums2 ，下标从 0 开始计数，其中nums1 是 nums2 的子集。
  对于每个 0 <= i < nums1.length ，找出满足 nums1[i] == nums2[j] 的下标 j ，并且在 nums2 确定 nums2[j] 的 下一个更大元素 。如果不存在下一个更大元素，那么本次查询的答案是 -1 。
  返回一个长度为 nums1.length 的数组 ans 作为答案，满足 ans[i] 是如上所述的 下一个更大元素 。

Answer:
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
            1. 先把 nums2 中的更大找到
            2. 映射一个 nums2 中值和上述结果的 map
            3. 根据 nums1 中的值，取上述 map 中的结果，作为最终结果
        """
        nums2_next_bigger = self.nextBigger(nums2)
        nums2_next_bigger_map = {}
        for i in range(len(nums2)):
            nums2_next_bigger_map[nums2[i]] = nums2_next_bigger[i]
        result = []
        for i in range(len(nums1)):
            result.append(nums2_next_bigger_map[nums1[i]])
        return result

    @staticmethod
    def nextBigger(nums: List[int]) -> List[int]:
        result = [0] * len(nums)
        reverse_result_stack = []
        for i in range(len(nums) - 1, -1, -1):
            # 把小的删了
            while reverse_result_stack and reverse_result_stack[-1] <= nums[i]:
                reverse_result_stack.pop()
            # 存储结果
            result[i] = reverse_result_stack[-1] if reverse_result_stack else -1
            # 把大的塞进来
            reverse_result_stack.append(nums[i])
        return result
```



```python
Leetcode index:
  739

Question:
  给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。

Answer:
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        next_bigger_index = self.nextBiggerIndex(temperatures)
        result = []
        for i in range(len(next_bigger_index)):
            result.append(max(int(next_bigger_index[i] - i), 0))
        return result

    @staticmethod
    def nextBiggerIndex(nums: List[int]) -> List[int]:
        result = [0] * len(nums)
        mid_stack = []
        for i in range(len(nums) - 1, -1, -1):
            while mid_stack and nums[mid_stack[-1]] <= nums[i]:
                mid_stack.pop()
            result[i] = mid_stack[-1] if mid_stack else -1
            mid_stack.append(i)
        return result
```



```python
Leetcode index:
  200

Question:
  给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
  岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
  此外，你可以假设该网格的四条边均被水包围。

Answer:
class Solution:
    def __init__(self):
        self.res = 0
        self.m = 0
        self.n = 0

    def numIslands(self, grid: List[List[str]]) -> int:
        self.m = len(grid)
        self.n = len(grid[0])

        for i in range(self.m):
            for j in range(self.n):
                if grid[i][j] == "1":
                    self.res += 1
                    self.travels(grid, i, j)

        return self.res

    def travels(self, grid: List[List[str]], i: int, j: int) -> None:
        if i < 0 or j < 0 or i >= self.m or j >= self.n:
            return
        if grid[i][j] == "0":
            return
        grid[i][j] = "0"
        self.travels(grid, i - 1, j)
        self.travels(grid, i + 1, j)
        self.travels(grid, i, j - 1)
        self.travels(grid, i, j + 1)
```



```python
Leetcode index:
  1254

Question:
  二维矩阵 grid 由 0 （土地）和 1 （水）组成。岛是由最大的4个方向连通的 0 组成的群，封闭岛是一个 完全 由1包围（左、上、右、下）的岛。
  请返回 封闭岛屿 的数目。

Answer:
class Solution:
    def __init__(self):
        self.m = 0
        self.n = 0
        self.res = 0

    def closedIsland(self, grid: List[List[int]]) -> int:
        self.m = len(grid)
        self.n = len(grid[0])
        for i in range(self.m):
            self.travels(grid, i, 0)
            self.travels(grid, i, self.n - 1)
        for j in range(self.n):
            self.travels(grid, 0, j)
            self.travels(grid, self.m - 1, j)
        for i in range(self.m):
            for j in range(self.n):
                if grid[i][j] == 0:
                    self.res += 1
                    self.travels(grid, i, j)
        return self.res

    def travels(self, grid: List[List[int]], i: int, j: int):
        if i < 0 or j < 0 or i >= self.m or j >= self.n:
            return

        if grid[i][j] == 1:
            return

        grid[i][j] = 1
        self.travels(grid, i - 1, j)
        self.travels(grid, i + 1, j)
        self.travels(grid, i, j - 1)
        self.travels(grid, i, j + 1)
```



```python
Leetcode index:
  1020

Question:
  给你一个大小为 m x n 的二进制矩阵 grid ，其中 0 表示一个海洋单元格、1 表示一个陆地单元格。
  一次 移动 是指从一个陆地单元格走到另一个相邻（上、下、左、右）的陆地单元格或跨过 grid 的边界。
  返回网格中 无法 在任意次数的移动中离开网格边界的陆地单元格的数量。

Answer:
class Solution:
    def __init__(self):
        self.m = 0
        self.n = 0
        self.res = 0

    def numEnclaves(self, grid: List[List[int]]) -> int:
        self.m = len(grid)
        self.n = len(grid[0])
        for i in range(self.m):
            self.travels(grid, i, 0)
            self.travels(grid, i, self.n - 1)
        for j in range(self.n):
            self.travels(grid, 0, j)
            self.travels(grid, self.m - 1, j)
        for i in range(self.m):
            for j in range(self.n):
                if grid[i][j] == 1:
                    self.res += 1
        return self.res

    def travels(self, grid: List[List[int]], i: int, j: int):
        if i < 0 or j < 0 or i >= self.m or j >= self.n:
            return

        if grid[i][j] == 0:
            return

        grid[i][j] = 0
        self.travels(grid, i - 1, j)
        self.travels(grid, i + 1, j)
        self.travels(grid, i, j - 1)
        self.travels(grid, i, j + 1)
```



```python
Leetcode index:
  695

Question:
  给你一个大小为 m x n 的二进制矩阵 grid 。
  岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在 水平或者竖直的四个方向上 相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。
  岛屿的面积是岛上值为 1 的单元格的数目。
  计算并返回 grid 中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。

Answer:
class Solution:
    def __init__(self):
        self.m = 0
        self.n = 0
        self.res = 0

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        self.m = len(grid)
        self.n = len(grid[0])
        for i in range(self.m):
            for j in range(self.n):
                if grid[i][j] == 1:
                    self.res = max(self.res, self.travels(grid, i, j))
        return self.res

    def travels(self, grid: List[List[int]], i: int, j: int) -> int:
        if i < 0 or j < 0 or i >= self.m or j >= self.n:
            return 0

        if grid[i][j] == 0:
            return 0

        grid[i][j] = 0

        return self.travels(grid, i - 1, j) + self.travels(grid, i + 1, j) + self.travels(grid, i, j - 1) + self.travels(grid, i, j + 1) + 1
```



```python
Leetcode index:
  1905

Question:
  给你两个 m x n 的二进制矩阵 grid1 和 grid2 ，它们只包含 0 （表示水域）和 1 （表示陆地）。一个 岛屿 是由 四个方向 （水平或者竖直）上相邻的 1 组成的区域。任何矩阵以外的区域都视为水域。
  如果 grid2 的一个岛屿，被 grid1 的一个岛屿 完全 包含，也就是说 grid2 中该岛屿的每一个格子都被 grid1 中同一个岛屿完全包含，那么我们称 grid2 中的这个岛屿为 子岛屿 。
  请你返回 grid2 中 子岛屿 的 数目 。

Answer:
class Solution:
    def __init__(self):
        self.m = 0
        self.n = 0
        self.res = 0

    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        self.m = len(grid1)
        self.n = len(grid1[0])

        for i in range(self.m):
            for j in range(self.n):
                if grid1[i][j] == 0 and grid2[i][j] == 1:
                    self.travels(grid2, i, j)

        for i in range(self.m):
            for j in range(self.n):
                if grid2[i][j] == 1:
                    self.res += 1
                    self.travels(grid2, i, j)

        return self.res

    def travels(self, grid: List[List[int]], i: int, j: int):
        if i < 0 or j < 0 or i >= self.m or j >= self.n:
            return

        if grid[i][j] == 0:
            return

        grid[i][j] = 0

        self.travels(grid, i + 1, j)
        self.travels(grid, i - 1, j)
        self.travels(grid, i, j + 1)
        self.travels(grid, i, j - 1)
```



```python
Leetcode index:
  1288

Question:
  给你一个区间列表，请你删除列表中被其他区间所覆盖的区间。
  只有当 c <= a 且 b <= d 时，我们才认为区间 [a,b) 被区间 [c,d) 覆盖。在完成所有删除操作后，请你返回列表中剩余区间的数目。

Answer:
class Solution:
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        intervals = sorted(intervals, key=lambda x: x[0])
        left = intervals[0][0]
        right = intervals[0][1]

        res = 0
        for i in range(1, len(intervals)):
            cur_interval = intervals[i]
            if left <= cur_interval[0] and right >= cur_interval[1]:
                res += 1
                print(res)
            if cur_interval[0] <= right <= cur_interval[1]:
                right = cur_interval[1]
            if right < cur_interval[0]:
                left = cur_interval[0]
                right = cur_interval[1]
        return len(intervals) - res
```



```python
Leetcode index:
  56

Question:
  以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。

Answer:
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals = sorted(intervals, key=lambda x: x[0])

        res = [intervals[0]]
        for i in range(1, len(intervals)):
            cur_interval = intervals[i]
            last_one = res[-1]
            if cur_interval[0] <= last_one[1]:
                last_one[1] = max(cur_interval[1], last_one[1])
            else:
                res.append(cur_interval)

        return res
```



```python
Leetcode index:
  986

Question:
  给定两个由一些 闭区间 组成的列表，firstList 和 secondList ，其中 firstList[i] = [starti, endi] 而 secondList[j] = [startj, endj] 。每个区间列表都是成对 不相交 的，并且 已经排序 。
  返回这 两个区间列表的交集 。
  形式上，闭区间 [a, b]（其中 a <= b）表示实数 x 的集合，而 a <= x <= b 。
  两个闭区间的 交集 是一组实数，要么为空集，要么为闭区间。例如，[1, 3] 和 [2, 4] 的交集为 [2, 3] 。

Answer:
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        i = 0
        j = 0

        res = []

        while i < len(firstList) and j < len(secondList):
            a1, a2 = firstList[i][0], firstList[i][1]
            b1, b2 = secondList[j][0], secondList[j][1]
            # 两个区间存在交集
            if b2 >= a1 and a2 >= b1:
                res.append([max(a1, b1), min(a2, b2)])
            if b2 < a2:
                j += 1
            else:
                i += 1
                
        return res
```



```python
Leetcode index:
  42

Question:
  给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

Answer:
class Solution:
    # 基本解法
    def trap(self, height: List[int]) -> int:
        res = 0
        for i in range(1, len(height) - 1):
            left_max = 0
            right_max = 0
            # 寻找右边最高的柱子
            for j in range(i, len(height)):
                right_max = max(height[j], right_max)
            # 寻找左边最高的柱子
            for j in range(i, -1, -1):
                left_max = max(height[j], left_max)
            res += min(left_max, right_max) - height[i]
        return res

    # 带有备忘录的解法
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0
        n = len(height)
        res = 0
        # 数组充当备忘录
        l_max = [0] * n
        r_max = [0] * n
        # 初始化 base case
        l_max[0] = height[0]
        r_max[n - 1] = height[n - 1]
        # 从左向右计算 l_max
        for i in range(1, n):
            l_max[i] = max(height[i], l_max[i - 1])
        # 从右向左计算 r_max
        for i in range(n - 2, -1, -1):
            r_max[i] = max(height[i], r_max[i + 1])
        # 计算答案
        for i in range(1, n - 1):
            res += min(l_max[i], r_max[i]) - height[i]
        return res

    # 双指针解法
    def trap(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        l_max, r_max = 0, 0
        res = 0
        while left < right:
            l_max = max(l_max, height[left])
            r_max = max(r_max, height[right])
            if l_max < r_max:
                res += l_max - height[left]
                left += 1
            else:
                res += r_max - height[right]
                right -= 1
        return res
```



```python
Leetcode index:
  300

Question:
  给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
  子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

Answer:
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1] * len(nums)

        for i in range(len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[j] + 1, dp[i])

        return max(dp)
```



```python
Leetcode index:
  53

Question:
  给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
  子数组 是数组中的一个连续部分。

Answer:
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        dp = [0] * len(nums)
        for i in range(len(nums)):
            dp[i] = max(nums[i], dp[i - 1] + nums[i])
        return max(dp)
```



```python
Leetcode index:
  1143

Question:
  给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0。
  一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
  例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
  两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。

Answer:
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m = len(text1)
        n = len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]
```



```python
Leetcode index:
  72

Question:
  给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数 。
  你可以对一个单词进行如下三种操作：
  	插入一个字符
    删除一个字符
    替换一个字符

Answer:
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m = len(word1)
        n = len(word2)

        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j - 1] + 1,
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1
                    )
        return dp[m][n]
```



```python
Leetcode index:
  416

Question:
  给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

Answer:
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        nums_sum = sum(nums)
        if nums_sum % 2 != 0:
            return False

        n = len(nums)
        nums_sum = nums_sum // 2

        # 定义二维数组 dp
        # dp[i][j] = x 表示，对于前 i 个物品（i 从 1 开始计数），当前背包的容量为 j 时，若 x 为 true，则说明可以恰好将背包装满，若 x 为 false，则说明不能恰好将背包装满
        dp = [[False] * (nums_sum + 1) for _ in range(n + 1)]

        # base case
        for i in range(n + 1):
            dp[i][0] = True

        for i in range(1, n + 1):
            for j in range(1, nums_sum + 1):
                if j - nums[i - 1] < 0:
                    # 背包容量不足，不能装入第 i 个物品
                    dp[i][j] = dp[i - 1][j]
                else:
                    # 装入或不装入背包
                    dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]
        return dp[n][nums_sum]
```



```python
Question:
  给你一个可装载重量为 W 的背包和 N 个物品，每个物品有重量和价值两个属性。其中第 i 个物品的重量为 wt[i]，价值为 val[i]，现在让你用这个背包装物品，最多能装的价值是多少？

Answer:
def knapsack(W: int, N: int, wt: List[int], val: List[int]) -> int:
    assert N == len(wt)
    # 初始化一个二维数组，用于存储状态
    # dp[i][j] 表示将前 i 个物品装入容量为 j 的背包中所获得的最大价值
    dp = [[0] * (W + 1) for _ in range(N + 1)]
    # 开始进行递推
    for i in range(1, N + 1):
        for w in range(1, W + 1):
            if w - wt[i - 1] < 0:
                # 当前商品 i 的重量已经超过了 w，无法被放入当前容量为 w 的背包中，只能选择不装入背包
                dp[i][w] = dp[i - 1][w]
            else:
                # 当前商品 i 的重量小于等于当前容量 w，可以尝试将其放入背包中
                # 取最大值，考虑是将其放入之前的最优方案中还是选择不放
                dp[i][w] = max(
                    dp[i - 1][w - wt[i - 1]] + val[i - 1],
                    dp[i - 1][w]
                )
    # 返回最大价值
    return dp[N][W]
```



```python
Question:
  快速排序

Answer:
def quick_sort(array, left, right):
    if left >= right:
        return
    low = left
    high = right
    key = array[low]
    while left < right:
        while left < right and array[right] > key:
            right -= 1
        array[left] = array[right]
        while left < right and array[left] <= key:
            left += 1
        array[right] = array[left]
    array[right] = key
    quick_sort(array, low, left - 1)
    quick_sort(array, left + 1, high)
```



```python
Question:
  归并排序

Answer:
def mergesort(seq):
    if len(seq) <= 1:
        return seq
    mid = len(seq) // 2  # 将列表分成更小的两个列表
    # 分别对左右两个列表进行处理，分别返回两个排序好的列表
    left = mergesort(seq[:mid])
    right = mergesort(seq[mid:])
    # 对排序好的两个列表合并，产生一个新的排序好的列表
    return merge(left, right)


def merge(left, right):
    """合并两个已排序好的列表，产生一个新的已排序好的列表"""
    result = []  # 新的已排序好的列表
    i = 0  # 下标
    j = 0
    # 对两个列表中的元素 两两对比。
    # 将最小的元素，放到result中，并对当前列表下标加1
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
    return result
```



```python
Leetcode index:
  215

Question:
  给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
  请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
  你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。

Answer:
class Solution(object):
    def findKthLargest(self, nums, k):
        low = 0
        high = len(nums) - 1
        while low <= high:
            mid = self.partition(nums, low, high)
            if mid == k - 1:
                return nums[mid]
            elif mid > k - 1:
                high = mid - 1
            else:
                low = mid + 1
        return -1

    @staticmethod
    def partition(nums, low, high):
        left = low
        right = high
        pivot = nums[low]

        while left < right:
            while left < right and nums[right] < pivot:
                right -= 1
            while left < right and nums[left] >= pivot:
                left += 1

            if left < right:
                tmp = nums[left]
                nums[left] = nums[right]
                nums[right] = tmp
        tmp = nums[low]
        nums[low] = nums[right]
        nums[right] = tmp
        return left
```



```python
Leetcode index:
  134

Question:
  在一条环路上有 n 个加油站，其中第 i 个加油站有汽油 gas[i] 升。
  你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。
  给定两个整数数组 gas 和 cost ，如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1 。如果存在解，则 保证 它是 唯一 的。

Answer:
# 如果选择站点i作为起点「恰好」无法走到站点j，那么i和j中间的任意站点k都不可能作为起点
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)
        diff_sum = 0
        for i in range(n):
            diff_sum += gas[i] - cost[i]

        if diff_sum < 0:
            return -1

        # 记录油箱油量
        tank = 0
        start = 0

        for i in range(n):
            tank += gas[i] - cost[i]
            # 无法从 start 走到 i
            if tank < 0:
                tank = 0
                start = i + 1

        if start == n:
            return 0
        else:
            return start
```



```python
Question:
  Kmeans

Answer:
import numpy as np
from matplotlib import pyplot


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, k=2, tolerance=0.0001, max_iter=300):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        self.centers_ = {}
        for i in range(self.k_):
            self.centers_[i] = data[i]

        for i in range(self.max_iter_):
            self.clf_ = {}
            for i in range(self.k_):
                self.clf_[i] = []
            # print("质点:",self.centers_)
            for feature in data:
                # distances = [np.linalg.norm(feature-self.centers[center]) for center in self.centers]
                distances = []
                for center in self.centers_:
                    # 欧拉距离
                    # np.sqrt(np.sum((features-self.centers_[center])**2))
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)

            # print("分组情况:",self.clf_)
            prev_centers = dict(self.centers_)
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis=0)

            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False
            if optimized:
                break

    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index


if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(k=2)
    k_means.fit(x)
    print(k_means.centers_)
    for center in k_means.centers_:
        pyplot.scatter(k_means.centers_[center][0], k_means.centers_[center][1], marker='*', s=150)

    for cat in k_means.clf_:
        for point in k_means.clf_[cat]:
            pyplot.scatter(point[0], point[1], c=('r' if cat == 0 else 'b'))

    predict = [[2, 1], [6, 9]]
    for feature in predict:
        cat = k_means.predict(predict)
        pyplot.scatter(feature[0], feature[1], c=('r' if cat == 0 else 'b'), marker='x')

    pyplot.show()
```

