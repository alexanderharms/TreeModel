```python
>>> #%matplotlib inline
... %load_ext Cython
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import sys
>>> from matplotlib import animation, rc
>>> from IPython.display import HTML
```

```python
>>> %%cython
... import numpy as np
... cimport numpy as np
... DTYPE = np.float_
... ctypedef np.float_t DTYPE_t
...
... class QuadTree:
...
...     def __init__(self, np.ndarray[np.float_t, ndim=2] poslist, np.ndarray[np.float_t, ndim=2] vellist, \
...                  double xmin, double ymin, double xmax, double ymax, double L, double dt, double G, int depth):
...         #assert poslist.dtype == DTYPE
...         #assert vellist.dtype == DTYPE
...
...         self.poslist = poslist
...         self.vellist = vellist
...         self.depth = depth
...         self.xmin = xmin
...         self.xmax = xmax
...         self.ymin = ymin
...         self.ymax = ymax
...         self.L = L
...         self.dt = dt
...         self.G = G
...
... #         if self.xmin is None:
... #             self.xmax, self.ymax = np.amax(poslist, axis=0)[1:3]
... #             self.xmin, self.ymin = np.amin(poslist, axis=0)[1:3]
...         #assert mins.dtype == DTYPE
...         #assert maxs.dtype == DTYPE
...         #assert sizes.dtype == DTYPE
...         #assert children.dtype == DTYPE
...         #assert mids.dtype == DTYPE
...         cdef float xmid, ymid
...         cdef np.ndarray[np.float_t, ndim=1] mins, maxs
...         cdef np.ndarray F
...
...         self.mins = np.asarray([self.xmin, self.ymin])
...         self.maxs = np.asarray([self.xmax, self.ymax])
...         self.sizes = self.maxs - self.mins
...         self.children = []
...         self.mids = (self.mins + self.maxs)/2
...         self.xmid, self.ymid = self.mids
...
...         self.CreateTree()
...         self.F = self.CalcTF()
...
...     def CreateTree(self):
...
...         cdef np.ndarray[np.float_t, ndim=2] q1, q2, q3, q4, q1vel, q2vel, q3vel, q4vel
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
...             if q1.shape[0] > 1:
...                 self.children.append(QuadTree(q1, q1vel, self.xmin, self.ymid, self.xmid, self.ymax, self.L, self.dt, \
...                                               self.G, self.depth-1))
...
...             if q2.shape[0] > 1:
...                 self.children.append(QuadTree(q2, q2vel, self.xmid, self.ymid, self.xmax, self.ymax, self.L, self.dt, \
...                                               self.G, self.depth-1))
...
...             if q3.shape[0] > 1:
...                 self.children.append(QuadTree(q3, q3vel, self.xmin, self.ymin, self.xmid, self.ymid, self.L, self.dt, \
...                                               self.G, self.depth-1))
...
...             if q4.shape[0] > 1:
...                 self.children.append(QuadTree(q4, q4vel, self.xmid, self.ymin, self.xmax, self.ymid, self.L, self.dt, \
...                                               self.G, self.depth-1))
...
...     def CalcF(self, np.ndarray[np.float_t, ndim=1] particle):
...         cdef int a
...         cdef float x, y
...         cdef float m
...         cdef np.ndarray[np.float_t, ndim=1] CM
...         #assert CM.dtype == DTYPE
...         cdef np.ndarray[np.float_t, ndim=1] CMrvec
...         cdef np.ndarray[np.float_t, ndim=1] CMrvecsq
...
...         #assert CMrvec.dtype == DTYPE
...         #assert CMrvecsq.dtype == DTYPE
...
...         cdef float CMrsqsum
...         cdef float CMr
...         cdef float F1x
...         cdef float F1y
...         cdef int ii
...         cdef int jj
...         cdef float sumposx
...         cdef float sumposy
...         global F1x, F1y
...
...         if self.sizes[0] == self.L:
...             F1x = 0
...             F1y = 0
...         if self.poslist.size != 0:
...             a, x, y, m = particle
...             sumposx = 0
...             sumposy = 0
...             summass = 0
...             for ii in range(len(self.poslist[:,0])):
...                 sumposx += self.poslist[ii, 3] * self.poslist[ii, 1]
...                 sumposy += self.poslist[ii, 3] * self.poslist[ii, 2]
...                 summass += self.poslist[ii, 3]
...             CM = np.asarray([sumposx / summass, sumposy / summass])
...             CMrvec = CM - [x, y] + 0.01
...             CMrvecsq = CMrvec**2
...             CMrsqsum = CMrvecsq[0] + CMrvecsq[1]
...             CMr = CMrsqsum**(0.5)
...             if (self.sizes[0]/CMr < 1) or (self.children==[]):
...                 F1x += (self.G * m * summass /(CMr**2)) * CMrvec[0]
...                 F1y += (self.G * m * summass /(CMr**2)) * CMrvec[1]
...             else:
...                 for jj in range(len(self.children)):
...                     self.children[jj].CalcF(particle)
...         return F1x, F1y
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
...     def MoveParticles(self):
...         cdef int k
...         cdef int k2
...         #cdef np.ndarray F
...         for k in range(self.poslist[:, 0].size):
...             # Calculate velocity, 1st step
...             self.vellist[k, 1] += 0.5 * self.F[k, 0] * self.dt
...             self.vellist[k, 2] += 0.5 * self.F[k, 1] * self.dt
...             # Calculate new positions
...             self.poslist[k, 1] += self.vellist[k, 1] * self.dt
...             self.poslist[k, 2] += self.vellist[k, 2] * self.dt
...             self.poslist[k, 1] = self.poslist[k, 1] % self.L
...             self.poslist[k, 2] = self.poslist[k, 2] % self.L
...
...         self.F = self.CalcTF()
...         for k2 in range(self.poslist[:, 0].size):
...             # Calculate velocity, 2nd step
...             self.vellist[k2, 1] += 0.5 * self.F[k2, 0] * self.dt
...             self.vellist[k2, 2] += 0.5 * self.F[k2, 1] * self.dt
...         self.CreateTree
...         return self.poslist
...
...     def Simulate(self):
...         #assert F.dtype == DTYPE
...         cdef int i
...         for i in range(100): # aantal tijdstappen
...             self.MoveParticles()
```

```python
>>> # Variables
... N = 128 # Number of particles
>>> L = 1
>>> dt = 1
>>> # Generate random positions
... ids = np.linspace(1, N, N)
>>> randpos = np.random.uniform(0, L, (N, 2))
>>> mass = np.ones(N)
>>> poslist = np.zeros((N, 4))
>>> vellist = np.zeros((N, 3))
>>> poslist[:, 0] = ids
>>> poslist[:, 1:3] = randpos[:,0:]
>>> poslist[:, 3] = mass
>>> vellist[:, 0] = ids
>>> vellist[:, 1:] = np.random.normal(0, np.sqrt(100), (N, 2))
>>> G = 6.64e-11
>>> # plt.figure()
... # plt.scatter(poslist[:, 1], poslist[:, 2])
... # plt.show()
```

```python
>>> # Plottin code
... a = QuadTree(poslist, vellist, 0, 0, L, L, L, dt, G, N)
...
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(0, L), ylim=(0, L))
>>> particles, = ax.plot([], [], 'bo', ms=6)
...
>>> def animate(i):
...     index, x, y, m = a.MoveParticles().T
...     particles.set_data(x, y)
...     return particles
...
>>> ani = animation.FuncAnimation(fig, animate)
>>> plt.show()
```

```python

```
