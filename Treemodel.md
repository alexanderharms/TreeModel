```python
>>> #%matplotlib inline
... %load_ext Cython
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from matplotlib import animation
The Cython extension is already loaded. To reload it, use:
  %reload_ext Cython
```

```python
>>> %%cython
... import numpy as np
... cimport numpy as np
... # DTYPE = np.float_
... # ctypedef np.float_t DTYPE_t
...
... class QuadTree:
...
...     def __init__(self, np.ndarray[np.float_t, ndim=2] poslist, np.ndarray[np.float_t, ndim=2] vellist, \
...                  double xmin, double ymin, double xmax, double ymax, double L, double dt, double G, int depth):
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
...         cdef float xmid, ymid
...         cdef np.ndarray[np.float_t, ndim=1] mins, maxs
...         cdef np.ndarray[np.float_t, ndim=2] F
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
...         cdef int a, ii, jj
...         cdef float x, y, m, CMRr, CMrsqsum, F1x, F1y, sumposx, sumposy
...         cdef np.ndarray[np.float_t, ndim=1] CM, CMrvec, CMrvecsq
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
...         cdef np.ndarray[dtype=np.float_t, ndim=2] F = np.zeros((self.poslist[:,0].size,2))
...         cdef int j
...
...         for j in range(self.poslist[:,0].size):
...             F[j,:] = self.CalcF(self.poslist[j,:])
...         self.F = F
...         return F
...
...     def MoveParticles(self):
...         cdef int k, k2
...
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
...     def Simulate(self, int nt):
...         cdef int i
...         for i in range(nt): # aantal tijdstappen
...             self.MoveParticles()
```

```python
>>> # Variables
... N = 128 # Number of particles
>>> L = 100 # [AU]
>>> dt = 1 # [years]
>>> # Generate random positions
... ids = np.linspace(1, N, N)
>>> #randpos = np.random.uniform(0, L, (N, 2))
... randr = np.random.uniform(-L/2, L/2, N)
>>> randtheta = np.random.uniform(0, 2*np.pi, N)
>>> randx = randr * np.cos(randtheta) + L/2
>>> randy = randr * np.sin(randtheta) + L/2
>>> randveltheta = np.random.normal(0, np.sqrt(0.1), N)
>>> randvelx = - randveltheta * np.sin(randtheta)
>>> randvely = randveltheta * np.cos(randtheta)
...
>>> mass = np.ones(N) # [Solar mass]
>>> poslist = np.zeros((N, 4))
>>> vellist = np.zeros((N, 3))
>>> poslist[:, 0] = ids
>>> #poslist[:, 1:3] = randpos[:,0:]
... poslist[:, 1] = randx
>>> poslist[:, 2] = randy
>>> poslist[:, 3] = mass
>>> vellist[:, 0] = ids
>>> #vellist[:, 1:] = np.random.normal(0, np.sqrt(100), (N, 2))#vellist[:, 1:] = np.random.normal(0, np.sqrt(10), (N, 2))
... vellist[:, 1] = randvelx
>>> vellist[:, 2] = randvely
>>> #G = 3.9e67 # [AU]**2/([Solar mass] * [year]**2)
... G = 1
>>> # plt.figure()
... # plt.scatter(poslist[:, 1], poslist[:, 2])
... # plt.xlim([0, L])
... # plt.ylim([0, L])
... # plt.show()
```

```python
>>> nt = 50
>>> a = QuadTree(poslist, veslist, 0, 0, L, L, L, dt, G, N)
>>> a.Simulate(nt)
```

```python
>>> # Plotting code
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
>>> def save_anim(file, title):
...     "saves the animation with a desired title"
...     Writer = animation.writers['ffmpeg']
...     writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
...     file.save(title + '.mp4', writer=writer)
...
>>> ani = animation.FuncAnimation(fig, animate, frames=10, repeat=False)
>>> save_anim(ani, 'Test')
>>> plt.show()
```

```python

```

```python

```
