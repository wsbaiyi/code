```python 

from collections import deque

class Tree:
    def __init__(self,val,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right

class Solution:
    # 递归遍历
    def recur(self,root):
        ans=[]

        # 后序
        def recur_1(root):
            if not root:
                return
            
            recur_1(root.left)
            recur_1(root.right)

            ans.append(root.val)
    
        recur_1(root)
        print(ans)
    # 迭代遍历
    def iter(self,root):

        if not root:
            return 
        que=deque([root])

        ans=[]

        while que:
            node=que.pop()
            if node:
                que.append(node)
                que.append(None)
                if node.right:
                    que.append(node.right)
                if node.left:
                    que.append(node.left)
            else:
                node=que.pop()
                ans.append(node.val)
        print(ans)

    # 递归层序遍历
    def recur_level(self,root):
        levels=[]


        def recur_level_1(root,level):
            if not root:
                return
            
            if len(levels)==level:
                levels.append([])
            levels[level].append(root.val)
            recur_level_1(root.left,level+1)
            recur_level_1(root.right,level+1)
        
        recur_level_1(root,0)
        print(levels)
    # 迭代层序
    def iter_level(self,root):

        if not root:
            return  
        que=deque([root])
        levels=[]
        level=-1
        while que:
            levels.append([])
            level+=1
            for i in range(len(que)):
                node=que.popleft()
                levels[level].append(node.val)
                if node.left:
                    que.append(node.left)
                if node.right:
                    que.append(node.right)
        
        print(levels)









```