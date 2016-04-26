```python
>>> %matplotlib inline
>>> %load_ext Cython
>>> import numpy as np
>>> import matplotlib.pyplot as plt
```

```python
>>> %%cython
... cimport numpy as np
... DTYPE = np.float_
... ctypedef np.float_t DTYPE_t
...
... class QuadTree:
...
...     def __init__(self, np.ndarray poslist, np.ndarray vellist, double xmin, double ymin, double xmax, double ymax, int depth=5):
...         assert poslist.dtype == DTYPE
...         assert vellist.dtype == DTYPE
...
...         self.poslist = poslist
...         self.vellist = vellist
...         self.depth = depth
...         self.xmin = xmin
...         self.xmax = xmax
...         self.ymin = ymin
...         self.ymax = ymax
...
... #         if self.xmin is None:
... #             self.xmax, self.ymax = np.amax(poslist, axis=0)[1:3]
... #             self.xmin, self.ymin = np.amin(poslist, axis=0)[1:3]
...         assert mins.dtype == DTYPE
...         assert maxs.dtype == DTYPE
...         assert sizes.dtype == DTYPE
...         assert children.dtype == DTYPE
...         assert mids.dtype == DTYPE
...         cdef float xmid
...         cdef float ymid
...
...         self.mins = [self.xmin, self.ymin]
...         self.maxs = [self.xmax, self.ymax]
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
...             q1vel = self.vellist[(self.poslist[:,1] < self.mids[0]) & (self.poslist[:,2] > self.mids[1])]
...             q2 = self.poslist[(self.poslist[:,1] > self.mids[0]) & (self.poslist[:,2] > self.mids[1])]
...             q2vel = self.vellist[(self.poslist[:,1] > self.mids[0]) & (self.poslist[:,2] > self.mids[1])]
...             q3 = self.poslist[(self.poslist[:,1] < self.mids[0]) & (self.poslist[:,2] < self.mids[1])]
...             q3vel = self.vellist[(self.poslist[:,1] < self.mids[0]) & (self.poslist[:,2] < self.mids[1])]
...             q4 = self.poslist[(self.poslist[:,1] > self.mids[0]) & (self.poslist[:,2] < self.mids[1])]
...             q4vel = self.vellist[(self.poslist[:,1] > self.mids[0]) & (self.poslist[:,2] < self.mids[1])]
...
...             if q1.shape[1] > 1:
...                 self.children.append(QuadTree(q1, q1vel, self.xmin, self.ymid, self.xmid, self.ymax, self.depth-1))
...
...             if q2.shape[1] > 1:
...                 self.children.append(QuadTree(q2, q2vel, self.xmid, self.ymid, self.xmax, self.ymax, self.depth-1))
...
...             if q3.shape[1] > 1:
...                 self.children.append(QuadTree(q3, q3vel, self.xmin, self.ymin, self.xmid, self.ymid, self.depth-1))
...
...             if q4.shape[1] > 1:
...                 self.children.append(QuadTree(q4, q4vel, self.xmid, self.ymin, self.xmax, self.ymid, self.depth-1))
...
...     def CalcF(self, np.ndarray particle):
...         cdef float L, dt, G
...         cdef int a
...         cdef float x, y
...         cdef np.ndarray CM
...         assert CM.dtype == DTYPE
...         cdef np.ndarray CMrvec
...         cdef np.ndarray CMrvecsq
...
...         assert CMrvec.dtype == DTYPE
...         assert CMrvecsq.dtype == DTYPE
...
...         cdef float CMrsqsum
...         cdef float CMr
...         cdef np.ndarray F1
...         assert F1.dtype == DTYPE
...         cdef int i
...         global F1
...
...         if self.sizes[0] == L:
...             F1 = 0
...         if self.poslist.size != 0:
...             a, x, y = particle
...             for i in range(len(self.poslist[:,0])):
...                 sumposx = 0
...                 sumposx += self.poslist[i, 1]
...                 sumposy = 0
...                 sumposy += self.poslist[i, 2]
...             CM = [sumposx /(self.poslist[:,1]).size, sumposy/(self.poslist[:,2]).size]
...             CMrvec = CM - [x, y] + 0.01
...             CMrvecsq = CMrvec**2
...             CMrsqsum = CMrvecsq[0] + CMrvecsq[1]
...             CMr = CMrsqsum**(0.5)
...             if (self.sizes[0]/CMr < 1) or (self.children==[]):
...                 F1 += (G/(CMr**2)) * CMrvec
...             else:
...                 for i in range(len(self.children)):
...                     self.children[i].CalcF(particle)
...         return F1
...
...
...     def CalcTF(self):
...         cdef np.ndarray F = np.zeros((self.poslist[:,0].size,2), dtype=DTYPE)
...         cdef int j
...
...         for j in range(self.poslist[:,0].size):
...             F[j,:] = self.CalcF(self.poslist[j,:])
...         self.F = F
...         return F
...
... #     def MoveParticles(self):
...
... #         # Verlet voor 1 stap
... #         for k in range(self.poslist[:,0].size):
...
...     def Simulate(self):
...         cdef float L, dt, G
...         cdef np.ndarray F
...         assert F.dtype == DTYPE
...         cdef int i
...         cdef int k
...         cdef int k2
...         F = self.CalcTF()
...         for i in range(1000): # aantal tijdstappen
...             for k in range(self.poslist[:, 0].size):
...                 # Calculate velocity, 1st step
...                 self.vellist[k, 1] += 0.5 * F[k, 0] * dt
...                 self.vellist[k, 2] += 0.5 * F[k, 1] * dt
...                 # Calculate new positions
...                 self.poslist[k, 1] += self.vellist[k, 1] * dt
...                 self.poslist[k, 2] += self.vellist[k, 2] * dt
...                 self.poslist[k, 1] = self.poslist[k, 1] % L
...                 self.poslist[k, 2] = self.poslist[k, 2] % L
...
...             F = self.CalcTF()
...             for k2 in range(self.poslist[:, 0].size):
...                 # Calculate velocity, 2nd step
...                 self.vellist[k2, 1] += 0.5 * F[k2, 0] * dt
...                 self.vellist[k2, 2] += 0.5 * F[k2, 1] * dt
... #             #self.MoveParticles()
...             self.CreateTree
```

```python
>>> # Variables
... N = 128 # Number of particles
>>> L = 10
>>> dt = 0.004
>>> # Generate random positions
... ids = np.linspace(1, N, N)
>>> randpos = np.random.uniform(0, L, (N, 2))
>>> poslist = np.zeros((N, 3))
>>> vellist = np.zeros((N, 3))
>>> poslist[:, 0] = ids
>>> poslist[:, 1:] = randpos[:,0:]
>>> vellist[:, 0] = ids
>>> vellist[:, 1:] = np.random.normal(0, np.sqrt(1), (N, 2))
>>> G = 6.64e-11
>>> plt.figure()
>>> plt.scatter(poslist[:, 1], poslist[:, 2])
>>> plt.show()
```

```python
>>> a = QuadTree(poslist, vellist, 0, 0, L, L)
>>> a.Simulate()
>>> plt.figure()
>>> plt.scatter(poslist[:, 1],poslist[:, 2])
>>> plt.show()
```

```python
>>> a = 1
>>> x = 2
>>> y = 3
>>> F1 = [x, y]
>>> F2 = [a, F1]
>>> print(F2)
```

```python

```
