```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> %matplotlib inline
```

```python
>>> class QuadTree:
...
...     def __init__(self, poslist, xmin, ymin, xmax, ymax, depth=5):
...
...         self.poslist = poslist
...         self.depth = depth
...         self.xmin = xmin
...         self.xmax = xmax
...         self.ymin = ymin
...         self.ymax = ymax
...
...         if self.xmin is None:
...             self.xmax, self.ymax = np.amax(poslist, axis=0)[1:3]
...             self.xmin, self.ymin = np.amin(poslist, axis=0)[1:3]
...
...         self.mins = np.asarray([self.xmin, self.ymin])
...         self.maxs = np.asarray([self.xmax, self.ymax])
...         self.sizes = self.maxs - self.mins
...         self.children = []
...         self.mids = (self.mins + self.maxs)/2
...         self.xmid, self.ymid = self.mids
...
...         self.CreateTree()
...
...     def CreateTree(self):
...         if self.depth > 0:
...
...             q1 = self.poslist[(self.poslist[:,1] < self.mids[0]) & (self.poslist[:,2] > self.mids[1])]
...             q2 = self.poslist[(self.poslist[:,1] > self.mids[0]) & (self.poslist[:,2] > self.mids[1])]
...             q3 = self.poslist[(self.poslist[:,1] < self.mids[0]) & (self.poslist[:,2] < self.mids[1])]
...             q4 = self.poslist[(self.poslist[:,1] > self.mids[0]) & (self.poslist[:,2] < self.mids[1])]
...
...             if q1.shape[1] > 1:
...                 self.children.append(QuadTree(q1, self.xmin, self.ymid, self.xmid, self.ymax, self.depth-1))
...
...             if q2.shape[1] > 1:
...                 self.children.append(QuadTree(q2, self.xmid, self.ymid, self.xmax, self.ymax, self.depth-1))
...
...             if q3.shape[1] > 1:
...                 self.children.append(QuadTree(q3, self.xmin, self.ymin, self.xmid, self.ymid, self.depth-1))
...
...             if q4.shape[1] > 1:
...                 self.children.append(QuadTree(q4, self.xmid, self.ymin, self.xmax, self.ymid, self.depth-1))
...
...     def CalcF(self, particle):
...
...         global F1
...         if self.sizes[0]==L:
...             F1 = 0
...         if self.poslist.size != 0:
...             a, x, y = particle
...             CM = np.asarray([np.sum(self.poslist, axis=0)[1]/(self.poslist[:,1]).size, np.sum(self.poslist, axis=0)[2]/(self.poslist[:,2]).size])
...             CMr = np.sqrt(np.sum((CM - np.asarray([x, y]))**2))
...             if (self.sizes[0]/CMr < 1) or (self.children==[]):
...                 F1 += 1
...             else:
...                 for i in range(len(self.children)):
...                     self.children[i].CalcF(particle)
...         return F1
...
...
...     def CalcTF(self):
...
...         F = np.zeros(self.poslist[:,0].size)
...         for j in range(self.poslist[:,0].size):
...             F[j] = self.CalcF(self.poslist[j,:])
...         self.F = F
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
>>> a = QuadTree(poslist, 0, 0, L, L)
>>> a.CalcTF()
>>> print(a.F)
```

```python

```
