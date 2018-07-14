# pratice lesson

## lesson 1

#### python introduction

* #coding=utf-8 可以用来规定文件编码
* 单引号和双引号的区别

* [x] homework
* [x] basic content

## lesson 2

#### python datastruct

* [x] homework
* [x] basic content

* list的切片和步长

```python
>>> l
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
>>> l[1:2]
[1]
>>> l[0:2]
[0, 1]
>>> l[::-1]
[99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
>>> l[0:99:10]
[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
```


* tuple的性质

```python
>>> t = 1, 2, 3
>>> t
(1, 2, 3)
>>> t[0]
1
>>> t[0] = 2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
```

* set的性质

```python
>>> l = [1, 2, 2, 3]
>>> s = set(l)
>>> s
{1, 2, 3}
```

* dict的性质

```python
>>> d = {}
>>> d[1] = 1
>>> d[2] = "2"
>>> d[3] = [0, 1, 2, 3]
>>> d
{1: 1, 2: '2', 3: [0, 1, 2, 3]}
>>> d.item()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'dict' object has no attribute 'item'
>>> d.items()
dict_items([(1, 1), (2, '2'), (3, [0, 1, 2, 3])])
>>> for key, val in d.items():
...     print(key, val)
... 
1 1
2 2
3 [0, 1, 2, 3]
```


* str的性质

```python
>>> s = "hello world!"
>>> s.split()
['hello', 'world!']
>>> " ".join(s.split())
'hello world!'
```

#### python lambda expr

* lambda表达式的用法

```python
>>> f = lambda x, y: x*x + y*y
>>> f
<function <lambda> at 0x7fce7df5fe18>
>>> f(3, 4)
25
```

* map的用法  

  在python 3中，map生成的是一个迭代器
```python
>>> l = [i for i in range(10)]
>>> l
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> map(lambda x:x**2, l)
<map object at 0x7fce7c59a198>
>>> for i in map(lambda x:x**2, l):
...     print(i)
... 
0
1
4
9
16
25
36
49
64
81
```

* filter的用法

```python
>>> a = [i for i in range(100) if i & 0x01 == 0]
>>> a
[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98]
>>> filter(lambda x: x%4, l)
>>> filter(lambda x: x%4, a)
<filter object at 0x7fad24c36048>
>>> for i in filter(lambda x: x%4, a):
...     print(i)
... 
2
6
10
14
18
22
26
30
34
38
42
46
50
54
58
62
66
70
74
78
82
86
90
94
98
```

* reduce的用法

```python
>>> a = [i for i in range(100) if i & 0x01 == 0]
>>> a
[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98]
>>> import functools
>>> functools.reduce(lambda x, y: x+y, a)
2450 
```

* eg: 使用map reduce统计每一个字母出现的个数

```python
>>> s = "hello world neu computer!"
>>> l = [i for i in list(s) if i != ' ']
>>> l
['h', 'e', 'l', 'l', 'o', 'w', 'o', 'r', 'l', 'd', 'n', 'e', 'u', 'c', 'o', 'm', 'p', 'u', 't', 'e', 'r', '!']

import funtools
def red(d, t):
    if t[0] in d:
        d[t[0]] += 1
    else:
        d[t[0]] = 1
    return d

functools.reduce(red, map(lambda x:(x, 1), list(s),d))
```

* 文件操作
* os库
* random库
* bs库

## lesson 3

* numpy速度比较块

```python
import time

import numpy as np

def normal():
    start = time.time()
    tot = 0
    l = [i for i in range(1, 1001)]
    for i in l:
        tot += i**2 + i**3

    stop = time.time()
    print("tot for res: ", tot)
    print("Time span: ", stop - start)


def np_method():
    start = time.time()
    tot = 0
    l = [i for i in range(1, 1001)]
    ln = np.array(l).astype(np.int64)
    tot += np.sum(ln**2)
    tot += np.sum(ln**3)
    print("tot for res: ", tot)
    print("Time span: ", time.time() - start)


if __name__ == "__main__":
    normal()
    np_method()
```

```
tot for res:  250834083500
Time span:  0.0004439353942871094
tot for res:  250834083500
Time span:  0.00015163421630859375
```

* 