```python
>>> %matplotlib inline
>>> %load_ext Cython
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import sys
>>> import scipy.stats as spstats
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
... #     def MoveParticles(self):
...
... #         # Verlet voor 1 stap
... #         for k in range(self.poslist[:,0].size):
...
...     def Simulate(self):
...         cdef np.ndarray F
...         #assert F.dtype == DTYPE
...         cdef int i
...         cdef int k
...         cdef int k2
...         F = self.CalcTF()
...         for i in range(10): # aantal tijdstappen
...             for k in range(self.poslist[:, 0].size):
...                 # Calculate velocity, 1st step
...                 self.vellist[k, 1] += 0.5 * F[k, 0] * self.dt
...                 self.vellist[k, 2] += 0.5 * F[k, 1] * self.dt
...                 # Calculate new positions
...                 self.poslist[k, 1] += self.vellist[k, 1] * self.dt
...                 self.poslist[k, 2] += self.vellist[k, 2] * self.dt
...                 self.poslist[k, 1] = self.poslist[k, 1] % self.L
...                 self.poslist[k, 2] = self.poslist[k, 2] % self.L
...
...             F = self.CalcTF()
...             for k2 in range(self.poslist[:, 0].size):
...                 # Calculate velocity, 2nd step
...                 self.vellist[k2, 1] += 0.5 * F[k2, 0] * self.dt
...                 self.vellist[k2, 2] += 0.5 * F[k2, 1] * self.dt
... #             #self.MoveParticles()
...             self.CreateTree
```

```python
>>> # Variables
... N = 128 # Number of particles
>>> L = 100 # [AU]
>>> dt = 1 # [years]
>>> a = 0.5
>>> #G = 3.9e67 # [AU]**2/([Solar mass] * [year]**2)
... G = 100
>>> Mh = 1
>>> ids = np.linspace(1, N, N)
...
>>> Pr = np.random.uniform(0, 1, N)
>>> r = a * np.sqrt(Pr) / (np.sqrt(Pr) - 1)
>>> # Von Neumann rejection for velocities
... Erand = np.random.uniform(-2, 0, N)
>>> fErand = np.random.uniform(0, 0.3, N)
...
>>> def hernquist(Erand):
...     q = np.sqrt(- a * Erand / (G * Mh))
...     fE = (np.sqrt(Mh)/(8*np.pi**3 * a ** (5/2) * np.sqrt(2*G)))  \
...     * (1/(1-q**2)**(5/2)) * (3*np.arcsin(q) + (q * (1 - q**2)**(1/2)) * (1 - 2*q**2) \
...     * (8 * q**4 - 8 * q**2 - 3))
...     return fE
...
>>> def potential(r):
...     potential = -G * Mh / (np.absolute(r) + a)
...     return potential
...
>>> fE = hernquist(Erand)
>>> fEcheck = fE > fErand
>>> fEcheck.astype(int)
>>> E = Erand[np.nonzero(Erand * fEcheck)]
>>> sumfEcheck = sum(fEcheck)
>>> while sumfEcheck < N:
...     Erand2 = np.random.uniform(-2, 0, (N - sum(fEcheck)))
...     fErand2 = np.random.uniform(0, 0.3, (N - sum(fEcheck)))
...     fE2 = hernquist(Erand2)
...     fEcheck2 = fE2 > fErand2
...     fEcheck2.astype(int)
...     E2 = Erand2[np.nonzero(Erand2 * fEcheck2)]
...     E = np.append(E, E2)
...     sumfEcheck += sum(fEcheck2)
...
>>> velmagsq = 2*(E[0:N] - potential(r))
...
>>> randvelx = np.random.uniform(-np.sqrt(np.absolute(velmagsq)), np.sqrt(np.absolute(velmagsq)), N)
>>> randvely = np.random.uniform(-np.sqrt(np.absolute(velmagsq - randvelx**2)), \
...                             np.sqrt(np.absolute(velmagsq - randvelx**2)), N)
...
>>> mass = (Mh * r**2) / (r + a)**2
>>> # Generate random positions
...
... #randpos = np.random.uniform(0, L, (N, 2))
... # randr = np.random.uniform(-L/2, L/2, N)
... randtheta = np.random.uniform(0, 2*np.pi, N)
>>> randphi = np.random.uniform(0, np.pi, N)
>>> randx = r * np.cos(randtheta) + L/2
>>> randy = r * np.sin(randtheta) + L/2
...
>>> # randveltheta = np.random.normal(0, np.sqrt(0.1), N)
... # randvelx = - randveltheta * np.sin(randtheta)
... # randvely = randveltheta * np.cos(randtheta)
...
... mass = np.ones(N) # [Solar mass]
>>> poslist = np.zeros((N, 4))
>>> vellist = np.zeros((N, 3))
>>> poslist[:, 0] = ids
>>> #poslist[:, 1:3] = randpos[:,0:]
... poslist[:, 1] = randx
>>> poslist[:, 2] = randy
>>> poslist[:, 3] = mass
...
...
>>> vellist[:, 0] = ids
>>> #vellist[:, 1:] = np.random.normal(0, np.sqrt(10), (N, 2))
... vellist[:, 1] = randvelx
>>> vellist[:, 2] = randvely
...
>>> plt.figure()
>>> plt.scatter(poslist[:, 1], poslist[:, 2])
>>> plt.xlim([0, L])
>>> plt.ylim([0, L])
>>> plt.show()
```

```python

```

```python
>>> a = QuadTree(poslist, vellist, 0, 0, L, L, L, dt, G, N)
>>> a.Simulate()
>>> plt.figure()
>>> plt.scatter(poslist[:, 1], poslist[:, 2])
>>> plt.xlim([0, L])
>>> plt.ylim([0, L])
>>> plt.show()
```

```python
>>> sys.getrecursionlimit()
```

```python

```
