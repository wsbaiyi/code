from collections  import deque
def main():
    m,n=map(int,input().split())
    graph=[[0]*n for _ in range(m)]
    visit=[[False]*n for _ in range(m)]

    for i in range(m):
        data=input().split()
        for j in range(n):
            graph[i][j]=int(data[j])
    

    direction={0:(-1,0),1:(0,1),2:(1,0),3:(0,-1)}
    ans=0
    for i in range(m):
        for j in range(n):
            if not visit[i][j] and graph[i][j]==1:
                visit[i][j]=True
                bfs(graph,visit,direction,i,j)
                ans+=1
    print(ans)
def dfs(graph,visit,direction,i,j):
    for key,value in direction.items():
        x,y=value
        now_x=i+x
        now_y=j+y

        if now_x>=0 and now_x<len(graph) and now_y>=0 and now_y<len(graph[0]) and not visit[now_x][now_y] and graph[now_x][now_y]==1:
            visit[now_x][now_y]=True
            dfs(graph,visit,direction,now_x,now_y)


def bfs(graph,visit,direction,i,j):
    que=deque([])
    que.append((i,j))
    while que:
        a,b=que.popleft()
        for key,value in direction.items():
            x,y=value
            now_x=a+x
            now_y=b+y

            if now_x>=0 and now_x<len(graph) and now_y>=0 and now_y<len(graph[0]) and not visit[now_x][now_y] and graph[now_x][now_y]==1:
                visit[now_x][now_y]=True
                que.append([now_x,now_y])
                bfs(graph,visit,direction,now_x,now_y)
if __name__=="__main__":
    main()
