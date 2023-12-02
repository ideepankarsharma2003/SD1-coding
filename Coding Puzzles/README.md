# Warmup

[Problem 01](https://www.metacareers.com/profile/coding_puzzles/?puzzle=513411323351554)
```python
# Write any import statements here

def getSum(A: int, B: int, C: int) -> int:
  # Write your code here
  return A+B+C
```

[Problem 02](https://www.metacareers.com/profile/coding_puzzles/?puzzle=1082217288848574)
```python
# Write any import statements here

def getWrongAnswers(N: int, C: str) -> str:
  # Write your code here
  ans= ''
  ans_d= {
    'A': 'B',
    'B': 'A'
  }
  for i in C:
    ans+=ans_d[i]
  return ans

```

[Problem 03](https://www.metacareers.com/profile/coding_puzzles/?puzzle=3641006936004915)
```python
from typing import List
# Write any import statements here

def getHitProbability(R: int, C: int, G: List[List[int]]) -> float:
  # Write your code here
  count=0
  for row in G:
    count+=  row.count(1)
  
  return count/(R*C)

```

