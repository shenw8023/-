
# 根据前序和中序遍历构造二叉树

class Solution:
    def __init__(self, preorder, inorder):
        # 构建字典用于在inorder中查找root索引
        self.inorder_root_index = {}
        for i, j in enumerate(inorder):
            self.inorder_root_index[j] = i
        preStart = 0
        preEnd = len(preorder)-1
        inStart = 0
        inEnd = len(inorder) - 1
        return self.build(preorder, preStart, preEnd,
                          inorder, inStart, inEnd) 
        
    
    def build(self, preorder, preStart, preEnd,
                    inorder, inStart, inEnd):
        "定义递归方法根据preorder和inorder片段构造二叉树，返回根结点"
        if preStart > preEnd:
            return None
        root = TreeNode(preorder[preStart]) #先序第一个为根
        root_index = self.inorder_root_index.get(preorder[preStart]) # 在中序中找到这个根结点，把中序分为左右子树
        left_size = root_index - inStart # 表示的是这个子树片段的长度，长度再加左起点就等于终点

        root.left = self.build(preorder, preStart+1, preStart+left_size,
                               inorder, inStart, root_index-1)
        root.right = self.build(preorder, preStart+left_size+1, preEnd,
                                inorder, root_index+1, inEnd)
        return root
        




# 剑指 Offer 26. 树的子结构
# 判断树B是否为A的子结构，子结构要求A中出现和B完全相同的结构和结点值
    # 遍历树 A 的所有节点，对 A 的所有节点做什么事呢？就是以 A 上的每个节点作为根节点，试图匹配树 B，也就是 compareTree 函数。
    
class Solution:
    # 遍历的思路理解
    def isSubStructure(self, A, B):
        if not A or not B:
            return False
        # 作为A的一个结点：
        # 1.如果A.val == B.val，则A就可以作为根结点尝试取匹配B
        if A.val == B.val and self.compare(A,B):
            return True

        # 2.如果A.val!=B.val，就继续递归遍历其子树是否包含B
        return self.isSubStructure(A.left, B) or self.isSubStructure(A.right,B)

    # 分解的思路理解：输入两个根节点，判断从A开始是否能完全匹配上B的所有结点
    def compare(self, A, B): 
        if B is None: # B为空，A肯定能匹配上B
            return True
        if B and not A: # A为空，B不空
            return False
        if A.val != B.val: #都不为空
            return False
        # 一对结点相等不能判断为True，要全部遍历完
        return self.compare(A.left, B.left) and self.compare(A.right, B.right)

        
# 剑指 Offer 27. 二叉树的镜像
    # 输出二叉树的镜像
    
class Solution:
    def inverse(self, root):
        "定义递归函数，将以root为根的树翻转，返回翻转后的树"
        if not root:
            return None
        left = self.inverse(root.left)
        right = self.inverse(root.right)
        
        # 神奇的后序位置
        root.left = right
        root.right = left
        return root
    
    
# 剑指 Offer 28. 对称的二叉树
    # 判断二叉树是否为对称的，即是否和他的镜像一样
class Solution:
    def isSymmetric(self, root) -> bool:
        return self.check(root, root)
    
    # 一棵树对称，要求其左右子树互为镜像
        # 什么叫两棵树互为镜像：根节点值要相等，one的左子树要和two的右子树互为镜像，右和左也是


    def check(self, root1, root2): # 定义递归函数，判断两个子树是否镜像对称
        if not root1 and not root2:
            return True
        if root1 and not root2:
            return False
        if root2 and not root1:
            return False
        return root1.val==root2.val and self.check(root1.left, root2.right) and self.check(root1.right, root2.left)
    
    
# 剑指 Offer 32 - I. 从上到下打印二叉树
    # 从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。
    
class Solution:
    def levelorder(self, root):
        "层序遍历"
        if not root:
            return []
        result = []
        queue = [root]
        while queue:
            sz = len(queue)
            for i in range(sz):
                node = queue.pop(0)
                result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return result
                

# 剑指 Offer 32 - II. 从上到下打印二叉树 II
    # 从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。  
    
class Solution:
    def levelorder(self, root):
        "层序遍历"
        if not root:
            return []
        result = []
        queue = [root]
        while queue:
            sz = len(queue)
            res_level = []
            for i in range(sz):
                node = queue.pop(0)
                res_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(res_level)
        return result        

# 剑指 Offer 32 - III. 从上到下打印二叉树 III
    # 按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。
    
class Solution:
    def levelorder(self, root):
        left = False  #注意要放在外面，第一次初始化
        if not root:
            return []
        result = []
        queue = [root]
        while queue:
            sz = len(queue)
            res_level = []
            left = not left # 每层变换方向
            for i in range(sz):
                node = queue.pop(0)
                res_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if left:
                result.append(res_level)
            else:
                result.append(res_level[::-1])
        return result        
        

# 剑指 Offer 33. 二叉搜索树的后序遍历序列
    # 给一个序列，判断其是否可能为某个二叉搜索树的后序遍历结果
    
class Solution:
    def verify(self, postorder):
        """二叉搜索树的后序结果，序列最后一个值为根结点，
        我们要划分左右子树，前面左子树（都小于根结点），后面右子树（都大于根结点）,
        那么第一个大于根节点的值必然是左右子树切分的位置（该位置的值属于右子树）"""
        return self.check(postorder, 0, len(postorder)-1)
        
        
    def check(self, postorder, i, j):
        "判断i到j这个子片段（子树）是否合法"
        if i>=j:
            return True
        p = i
        while postorder[i] < postorder[j]:
            p += 1
        m = p #记录这个位置（m的值是属于右子树的）
        # 以上保证左子树都小于根结点
        # 一下保证右子树都大于根节点
        while postorder[p] > postorder[j]:
            p += 1
        return p == j and self.check(postorder, i, m-1) and self.check(postorder, m, j-1)
            
        
# 剑指 Offer 34. 二叉树中和为某一值的路径
    # 找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

class Solution:
    def pathSum(self, root, target):
        self.result = []
        self.travel(root, target, [])
        return self.result
        
    def travel(self, root, target, path):
        "回溯记得撤销选择，递归遍历，在满足要求的时候保存结果"
        if not root:
            return
        if root.val == target and not root.left and not root.right:
            self.result.append(path + [root.val])
            
        path.append(root.val)
        self.travel(root.left, target-root.val, path)
        self.travel(root.right, target-root.val, path)
        path.pop()
        
        

# 剑指 Offer 37. 序列化和反序列化二叉树

class Solution:
    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return ""
        self.null = "#"
        res = []
        self.se(root, res)
        return ",".join(res)

    def se(self, root, res):
        if not root:
            res.append(self.null)
            return
        res.append(str(root.val))
        self.se(root.left, res)
        self.se(root.right, res)

        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return None
        res = self.de(data.split(","))
        return res
    
    def de(self, data_list):
        if not data_list:
            return None
        root_val = data_list.pop(0)
        if root_val == self.null:
            return None

        root = TreeNode(int(root_val))

        root.left = self.de(data_list)
        root.right = self.de(data_list)
        return root


"""
涉及序列，并且子序列对应的子树 都要做判断的题目，
例如根据二叉树前序和中序结果还原二叉树；判断一个序列是否可能为二叉搜索树的后续结果；
都定义递归函数判断子结构，参数i,j分别指定起止位置。
"""