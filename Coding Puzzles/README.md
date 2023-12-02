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


# Level 01

[Problem 01](https://www.metacareers.com/profile/coding_puzzles/?puzzle=203188678289677)
Not working yet !!!
```python
from typing import List
# Write any import statements here

def can_sit(arr):
  if True not in arr:
    return True
  else:
    return False

def getMaxAdditionalDinersCount(N: int, K: int, M: int, S: List[int]) -> int:
  # Write your code here
  # N seats ---> 1 to N
  
  # K seats --- diner --- K seats
  
  diner_seats= [False]*N
  for i in S:
    diner_seats[i-1]=True
  count=0
  for i in range(K, N):
    if can_sit(diner_seats[i-K: i+K+1]):
      diner_seats[i]=1
      count+=1
    
  
  print(diner_seats)
  return count

```



<!--
[Problem 01]
```python
```

[Problem 01]
```python
```

[Problem 01]
```python
```

[Problem 01]
```python
```
-->

