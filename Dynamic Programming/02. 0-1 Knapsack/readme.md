# 0-1 Knapsack


- [0 - 1 Knapsack Problem](https://practice.geeksforgeeks.org/problems/0-1-knapsack-problem0945/1)

**Recursive** <br>
```python

class Solution:
    
    #Function to return max value that can be put in knapsack of capacity W.
    def knapSack(self,W, wt, val, n):
       
        if n==0 or W==0:
            max_profit= 0
            return max_profit
        
        if wt[n-1]>W:
            # can't include in knapsack
            max_profit= self.knapSack(W, wt, val, n-1)
            return max_profit
        else:
            max_profit= max(
                self.knapSack(W, wt, val, n-1), # don't include
                val[n-1]+self.knapSack(W-wt[n-1], wt, val, n-1) # include 
            )
            return max_profit

```

