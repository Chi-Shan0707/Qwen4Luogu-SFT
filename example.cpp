#include <iostream>
#include <cstring>
#include <cstdio>
using namespace std;

const int MAXN = 500000 + 1;
int f[MAXN], g[MAXN];
int t, n;
char a[MAXN];

int main() {
    cin >> t;
    while (t--) {
        cin >> n;
        cin >> a;
        
        // 初始化前缀和数组
        memset(f, 0, sizeof(f));
        memset(g, 0, sizeof(g));
        
        // 计算前缀和：f[i] 表示前i位的和
        for (int i = 1; i <= n; i++) {
            f[i] = f[i - 1] + (a[i - 1] == '1');
        }
        
        // 计算后缀和：g[i] 表示从i位开始的和
        for (int i = n - 1; i >= 1; i--) {
            g[i] = g[i + 1] + (a[i] == '1');
        }
        
        // 从左到右遍历，计算子序列和的和
        int ans = 0;
        for (int i = 1; i < n; i++) {
            // 计算前缀和 + 后缀和
            int sum = f[i] + g[i + 1];
            // 如果前缀和为0，那么后缀和就是子序列和
            if (f[i] == 0) {
                ans += sum;
            } else {
                // 如果前缀和不为0，那么需要考虑是否交换位置
                int swap_sum = g[i] + f[i - 1];
                ans += min(sum, swap_sum);
            }
        }
        
        // 输出结果
        printf("%d\n", ans);
    }
    return 0;
}