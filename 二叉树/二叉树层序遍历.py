import sys
from collections import deque

# 1. 定义二叉树节点 (平台不提供时需要自己写)
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 2. 核心算法部分
class Solution:
    def level_stack(self, root: TreeNode) -> list[list[int]]:
        res=[]
        st=deque()
        if not root:
            return st
        # deque 的构造函数接受的是一个可迭代对象（Iterable）。所以要加[]把root包起来
        st.append([root])
        while st:
            level=[]
            size=len(st)
            for _ in range(size):
                node=st.popleft()

                if node.left:
                    st.append(node.left)
                if node.right:
                    st.append(node.right)
                level.append(node.val)
            res.append(level)
        return res
    def level_recursion(self, root: TreeNode) -> list[list[int]]:
        print(';')
        
        levels=[]

        def dfs(node, level):
            if not node:
                return
            if level==len(levels):
                levels.append([])
            
            levels[level].append(node.val)
            dfs(node.left,level+1)
            dfs(node.right,level+1)
        dfs(root,0)
        return levels



# 3. 辅助函数：根据数组构建二叉树 (ACM模式的重点)
def build_tree(data):
    # 如果输入为空，或者根节点就是 null
    if not data or data[0] == 'null':
        return None
    
    root = TreeNode(int(data[0]))
    queue = deque([root])
    i = 1
    
    while queue and i < len(data):
        node = queue.popleft()
        
        # 尝试构建左孩子
        if i < len(data) and data[i] != 'null':
            node.left = TreeNode(int(data[i]))
            queue.append(node.left)
        i += 1
        
        # 尝试构建右孩子
        if i < len(data) and data[i] != 'null':
            node.right = TreeNode(int(data[i]))
            queue.append(node.right)
        i += 1
        
    return root

# 4. 主函数：处理标准输入和输出
def main():
    # sys.stdin 适合处理可能有多行测试用例的情况
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        # 解析字符串："[1,null,2,3]" -> ['1', 'null', '2', '3']
        clean_line = line.strip('[]')
        # '1,2,3,4,5,null,8,null,null,6,7,9'
        print(f"输入的原始字符串: {clean_line}")
        if not clean_line.strip():
            data = []
        else:
            # 去除可能存在的空格并分割
            # ['1', 'null', '2', '3']
            data = [x.strip() for x in clean_line.split(',')]
        
        # 构建二叉树
        root = build_tree(data)
        
        # 求解
        solution = Solution()
        res = solution.level_stack(root)
        
        # 格式化输出，例如将列表 [1, 2, 3] 转为字符串 "[1,2,3]"
        
        print(res)

if __name__ == '__main__':
    main()