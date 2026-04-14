import sys

def main():
    # 1. 修正读取方式：sys.stdin.read().split() 能更好地处理换行和多余空格
    data = sys.stdin.read().split()
    if not data:
        return
    
    # 将所有字符串转化为整数
    data = list(map(int, data))
    
    # 2. 结构化赋值
    # 假设输入格式为: n bagweight / weight列表 / value列表
    n, bagweight = data[0], data[1]
    weight = data[2 : 2 + n]
    value = data[2 + n : 2 + 2 * n]

    # 初始化 dp 数组
    dp = [0] * (bagweight + 1)

    # 3. 核心逻辑（这里你的逻辑是正确的，但要注意 range 的边界）
    for i in range(n):
        # 倒序遍历，确保每个物品只取一次
        # range(start, stop, step) 停止位置不包含 weight[i]-1，即到 weight[i] 为止
        for j in range(bagweight, weight[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])

    print(dp[bagweight])

if __name__ == '__main__':
    main()