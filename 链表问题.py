
#[ ]很重要  迭代法反转一个链表 
# 反转以 a 为头结点的链表
def reverse(a: ListNode) -> ListNode:
    pre, cur = None, a
    while cur:
        nxt = cur.next
        # 逐个结点反转
        cur.next = pre
        # 更新指针位置
        pre = cur  #这两句顺序不能错，因为cur被依赖 
        cur = nxt
    # 返回反转后的头结点
    return pre



# 反转区间 [a, b) 的元素，注意是左闭右开
def reverse(a:ListNode, b:ListNode) -> ListNode:
    pre, cur, nxt = None, a, a
    # while  终止的条件改一下就行了
    while cur != b:
        nxt = cur.next
        cur.next = pre
        pre = cur
        cur = nxt
    # 返回反转后的头结点
    return pre


# 递归反转链表
#25 K个一组反转链表
# 回文链表