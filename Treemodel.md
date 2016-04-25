```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> %matplotlib inline
```

```python
>>> class QuadTree:
...     def __init__(self, poslist, xmin, ymin, xmax, ymax, depth=1):
...
...         if xmin is None:
...             xmax, ymax = np.amax(poslist, axis=0)[1:3]
...             xmin, ymin = np.amin(poslist, axis=0)[1:3]
...
...         self.mins = np.asarray([xmin, ymin])
...         self.maxs = np.asarray([xmax, ymax])
...         self.sizes = self.maxs - self.mins
...         self.children = []
...         mids = (self.mins + self.maxs)/2
...         xmid, ymid = mids
...
...         if depth > 0:
...
...             q1 = poslist[(poslist[:,1] < mids[0]) & (poslist[:,2] > mids[1])]
...             q2 = poslist[(poslist[:,1] > mids[0]) & (poslist[:,2] > mids[1])]
...             q3 = poslist[(poslist[:,1] < mids[0]) & (poslist[:,2] < mids[1])]
...             q4 = poslist[(poslist[:,1] > mids[0]) & (poslist[:,2] < mids[1])]
...
...             if q1.shape[1] > 1:
...                 self.children.append(QuadTree(q1, xmin, ymid, xmid, ymax, depth-1))
...
...             if q2.shape[1] > 1:
...                 self.children.append(QuadTree(q2, xmid, ymid, xmax, ymax, depth-1))
...
...             if q3.shape[1] > 1:
...                 self.children.append(QuadTree(q3, xmin, ymin, xmid, ymid, depth-1))
...
...             if q4.shape[1] > 1:
...                 self.children.append(QuadTree(q4, xmid, ymin, xmax, ymid, depth-1))
```

```python
>>> def Split4(poslist, xmin=None, ymin=None, xmax=None, ymax=None):
...
...     if xmin is None:
...         xmax, ymax = np.amax(poslist, axis=0)[1:3]
...         xmin, ymin = np.amin(poslist, axis=0)[1:3]
...
...     q1 = poslist[(poslist[:,1] < (xmin+xmax)/2) & (poslist[:,2] > (ymin+ymax)/2)]
...     q2 = poslist[(poslist[:,1] > (xmin+xmax)/2) & (poslist[:,2] > (ymin+ymax)/2)]
...     q3 = poslist[(poslist[:,1] < (xmin+xmax)/2) & (poslist[:,2] < (ymin+ymax)/2)]
...     q4 = poslist[(poslist[:,1] > (xmin+xmax)/2) & (poslist[:,2] < (ymin+ymax)/2)]
...
...     return q1, q2, q3, q4
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
... N = 128 # Number of particles
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
>>> plt.figure()
>>> plt.scatter(poslist[:,1],poslist[:,2])
>>> plt.show()
```

```python
>>> q1 = Split4(poslist, 0, 0, L, L)[1]
>>> plt.figure()
>>> plt.scatter(q1[:,1],q1[:,2])
>>> plt.xlim([0,10])
>>> plt.ylim([0,10])
>>> plt.show()
```

```python
>>> q11 = Split4(q1, L/2, L/2, L, L)[1]
>>> plt.figure()
>>> plt.scatter(q11[:,1],q11[:,2])
>>> plt.xlim([0,10])
>>> plt.ylim([0,10])
>>> plt.show()
```

```python
>>> QuadTree(poslist, 0, 0, L, L)
<__main__.QuadTree at 0x13b4554cc50>
```

```python

```
