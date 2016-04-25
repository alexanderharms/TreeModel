```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> %matplotlib inline
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
...         for i in range(3):
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
>>> class QuadTree:
...
...     def __init__(self, poslist, xmin, ymin, xmax, ymax, depth=5):
...
...         self.poslist = poslist
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
...
...     def CalcF(self):
...         global F
...         if self.sizes[0]==L:
...             F = 0
...         if self.poslist.size != 0:
...             a, x, y = self.poslist[0,:]
...             CM = np.asarray([np.sum(self.poslist, axis=0)[1]/(self.poslist[:,1]).size, np.sum(self.poslist, axis=0)[2]/(self.poslist[:,2]).size])
...             CMr = np.sqrt(np.sum((CM - np.asarray([x, y]))**2))
...             if (self.sizes[0]/CMr < 1) or (self.children==[]):
...                 F += 1
...             else:
...                 for i in range(len(self.children)):
...                     self.children[i].CalcF()
```

```python
>>> a=QuadTree(poslist, 0, 0, L, L)
```

```python
>>> print(a.CalcF())
>>> print(F)
>>> print(zz)
None
122
0
C:\Users\skros\Anaconda3\lib\site-packages\ipykernel\__main__.py:45: RuntimeWarning: divide by zero encountered in double_scalars
```

```python
...     def CalcF(self):
...         global F
...         if self.sizes[0]==L:
...             F = 0
...
...         if self.poslist.size == 0:
...             return
...         else:
...             a, x, y = self.poslist[0,:]
...             CM = np.asarray([np.sum(self.poslist, axis=0)[1]/(self.poslist[:,1]).size, np.sum(self.poslist, axis=0)[2]/(self.poslist[:,2]).size])
...             CMr = np.sqrt(np.sum((CM - np.asarray([x, y]))**2))
...             if (self.sizes[0]/CMr < 1):# or (self.children==[]):
...                 F += 1
...             else:
...                 global zz
...                 zz = 0
...                 for i in range(len(self.children)):
...                     zz += 1
...                     self.children[i].CalcF()
```
