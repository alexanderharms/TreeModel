```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> %matplotlib inline
```

```python
>>> # def Split4(poslist, xmin=None, ymin=None, xmax=None, ymax=None):
...
... #     if xmin is None:
... #         xmax, ymax = np.amax(poslist, axis=0)[1:3]
... #         xmin, ymin = np.amin(poslist, axis=0)[1:3]
...
... #     q1 = poslist[(poslist[:,1] < (xmin+xmax)/2) & (poslist[:,2] > (ymin+ymax)/2)]
... #     q2 = poslist[(poslist[:,1] > (xmin+xmax)/2) & (poslist[:,2] > (ymin+ymax)/2)]
... #     q3 = poslist[(poslist[:,1] < (xmin+xmax)/2) & (poslist[:,2] < (ymin+ymax)/2)]
... #     q4 = poslist[(poslist[:,1] > (xmin+xmax)/2) & (poslist[:,2] < (ymin+ymax)/2)]
...
... #     return q1, q2, q3, q4
```

```python
>>> def Create_tree(poslist, xmin=None, ymin=None, xmax=None, ymax=None, depth):
...
...
...     if xmin is None:
...         xmax, ymax = np.amax(poslist, axis=0)[1:3]
...         xmin, ymin = np.amin(poslist, axis=0)[1:3]
...
...     q1, q2, q3, q4 = Split4(poslist, 0, 0, L, L)
...     if q1[1].shape <= 1 & q2[1].shape <= 1 & q3[1].shape <= 1 & q4[1].shape <= 1
...         return q1, q2, q3, q4
...     else
...         for i in range(4):
```

```python
>>> # Variables
... N = 64 # Number of particles
>>> L = 10
...
>>> # Generate random positions
... ids = np.linspace(1, N, N)
>>> randpos = np.random.uniform(0, L, (N, 2))
>>> poslist = np.zeros((N, 3))
>>> poslist[:, 0] = ids
>>> poslist[:, 1:] = randpos[:,0:]
```

```python
>>> def Split4(q, poslist, boxlist, children, boxnum, i):
...     xmin = boxlist[boxnum, 2]
...     xmax = boxlist[boxnum, 3]
...     ymin = boxlist[boxnum, 4]
...     ymax = boxlist[boxnum, 5]
...     a = poslist[(poslist[:,1] < (xmin+xmax)/2) & (poslist[:,2] > (ymin+ymax)/2)]
...     b = poslist[(poslist[:,1] > (xmin+xmax)/2) & (poslist[:,2] > (ymin+ymax)/2)]
...     c = poslist[(poslist[:,1] < (xmin+xmax)/2) & (poslist[:,2] < (ymin+ymax)/2)]
...     d = poslist[(poslist[:,1] > (xmin+xmax)/2) & (poslist[:,2] < (ymin+ymax)/2)]
...     q[i*4 + 0, 0:len(a), :] = a
...     q[i*4 + 1, 0:len(b), :] = b
...     q[i*4 + 2, 0:len(c), :] = c
...     q[i*4 + 3, 0:len(d), :] = d
...     boxlist[i*4 + 0, 2:6] = [xmin, (xmin+xmax)/2, (ymin+ymax)/2, ymax]
...     boxlist[i*4 + 1, 2:6] = [(xmin+xmax)/2, xmax, (ymin+ymax)/2, ymax]
...     boxlist[i*4 + 2, 2:6] = [xmin, (xmin+xmax)/2, ymin, (ymin+ymax)/2]
...     boxlist[i*4 + 3, 2:6] = [(xmin+xmax)/2, xmax, ymin, (ymin+ymax)/2]
...     children[boxnum, :] = np.linspace(0,4,4) + i*4
...
...     return q, boxlist, children
```

```python
>>> q = np.zeros((N**2,N,3))
>>> boxlist = np.zeros((N**2,6))
>>> children = np.zeros((N**2, 4))
>>> boxlist[0, 2:6] = [0, L, 0, L]
>>> q, boxlist, children = Split4(q, poslist, boxlist, children, 0, 0)
>>> check = q[:, :, 0] > 0
>>> check2 = np.sum(check, axis=1)
>>> i = 1
>>> while any(check2 > 1):
...     check3 = check2 > 1
...     r = np.array(range(len(check3)))
...     c = r[check3]
...     d = 0
...     for d in range(len(c)):
...         q, boxlist, children = Split4(q, q[c[d],:,:], boxlist, children, c[d], i+d)
...
...     i =+ (d + 1)
...     check = q[:,:,0] > 0
...     check2 = np.sum(check[i*4:], axis=1)
...
>>> k = np.amax(np.nonzero(q[:,0,0]))
>>> l = np.amax(np.nonzero(boxlist[:,0]))
>>> m = np.amax(np.nonzero(children[:,0]))
>>> # while sum(q[-1, :, 0] > 0) > 1:
... #     i = 0
... #     for m in range(i,i+1):
... #         for n in range(i,i+1):
... #             q = Split4(q, poslist, m*L/(i+1), n*L/(i+1), (m+1)*L/(i+1), (n+1)*L/(i+1), (m+1)+(n+1))
... #     i =+ 1
... # print(q)
```

```python
>>> boxlist[0, 2:6] = [0, L, 0, L]
>>> boxlist[0,:]
array([  0.,   0.,   0.,  10.,   0.,  10.])
```

```python
>>> plt.figure()
>>> plt.scatter(poslist[:,1],poslist[:,2])
>>> plt.show()
```

```python
>>> q1 = Split4(poslist, 0, 0, L, L)[0]
>>> plt.figure()
>>> plt.scatter(q1[:,1],q1[:,2])
>>> plt.xlim([0,10])
>>> plt.ylim([0,10])
>>> plt.show()
```

```python
>>> q11 = Split4(q1, 0, L/2, L/2, L)[0]
>>> plt.figure()
>>> plt.scatter(q11[:,1],q11[:,2])
>>> plt.xlim([0,10])
>>> plt.ylim([0,10])
>>> plt.show()
```

```python
>>> q[:, 0, -1] > 0
array([False, False, False, False], dtype=bool)
```

```python
>>> sum(q[:, 0, -1] > 0)
0
```

```python
>>> xmin = 0
>>> xmax = L
>>> ymin = 0
>>> ymax = L
>>> poslist[(poslist[:,1] < (xmin+xmax)/2) & (poslist[:,2] > (ymin+ymax)/2)]
array([[ 1.        ,  2.53410143,  6.21508753]])
```

```python
>>> poslist[(poslist[:,1] > (xmin+xmax)/2) & (poslist[:,2] > (ymin+ymax)/2)]
array([[ 2.        ,  9.730974  ,  5.16101999],
       [ 4.        ,  5.11417445,  7.63920153]])
```

```python

```
