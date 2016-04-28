```python
>>> #%matplotlib inline
... %load_ext Cython
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from matplotlib import animation
>>> import mpl_toolkits.mplot3d.axes3d as p3
>>> import sys
The Cython extension is already loaded. To reload it, use:
  %reload_ext Cython
```

```python
>>> %%cython
... #import numpy as np
... #cimport numpy as np
... # DTYPE = np.float_
... # ctypedef np.float_t DTYPE_t
...
... cdef class OctoTree:
...     cdef public poslist, vellist, depth, xmin, xmax, ymin, ymax, zmin, zmax, L, dt, G
...     cdef public sizesx, sizesy, sizesz, children, xmid, ymid, zmid, F
...     def __init__(self, list poslist, list vellist, \
...                  double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, \
...                  double L, double dt, double G, int depth):
...
...         self.poslist = poslist
...         self.vellist = vellist
...         self.depth = depth
...         self.xmin = xmin
...         self.xmax = xmax
...         self.ymin = ymin
...         self.ymax = ymax
...         self.zmin = zmin
...         self.zmax = zmax
...         self.L = L
...         self.dt = dt
...         self.G = G
...
...         cdef double xmid, ymid, zmid
...         cdef double F[256][2]
...         cdef double sizesx, sizesy, sizesz
...
...         self.sizesx = self.xmax - self.xmin
...         self.sizesy = self.ymax - self.ymin
...         self.sizesz = self.zmax - self.zmin
...         self.children = []
...         self.xmid = (self.xmin + self.xmax)/2
...         self.ymid = (self.ymin + self.ymax)/2
...         self.zmid = (self.zmin + self.zmax)/2
...
...         self.CreateTree()
...         self.F = self.CalcTF()
...
...     def CreateTree(self):
...
...         cdef double q1[1][2]
...         cdef double q2[1][2]
...         cdef double q3[1][2]
...         cdef double q4[1][2]
...         cdef double q5[1][2]
...         cdef double q6[1][2]
...         cdef double q7[1][2]
...         cdef double q8[1][2]
...         cdef double q1vel[1][2]
...         cdef double q2vel[1][2]
...         cdef double q3vel[1][2]
...         cdef double q4vel[1][2]
...         cdef double q5vel[1][2]
...         cdef double q6vel[1][2]
...         cdef double q7vel[1][2]
...         cdef double q8vel[1][2]
...
...         cdef int qq1
...         if self.depth > 0:
...             for qq1 in range(len(self.poslist)):
...                 if (self.poslist[qq1][0] < self.xmid) & (self.poslist[qq1][1] > self.ymid) & (self.poslist[qq1][2] < self.zmid):
...                     if qq1 == 0:
...                         q1[qq1][0] = self.poslist[qq1][0]
...                         q1[qq1][1] = self.poslist[qq1][1]
...                         q1[qq1][2] = self.poslist[qq1][2]
...                         q1vel[qq1][0] = self.vellist[qq1][0]
...                         q1vel[qq1][1] = self.vellist[qq1][1]
...                         q1vel[qq1][2] = self.vellist[qq1][2]
...                     else:
...                         q1.append([self.poslist[qq1][0], self.poslist[qq1][1], self.poslist[qq1][2]])
...                         q1vel.append([self.vellist[qq1][0], self.vellist[qq1][1], self.vellist[qq1][2]])
...                 elif (self.poslist[qq1][0] > self.xmid) & (self.poslist[qq1][1] > self.ymid) & (self.poslist[qq1][2] < self.zmid):
...                     if qq1 == 0:
...                         q2[qq1][0] = self.poslist[qq1][0]
...                         q2[qq1][1] = self.poslist[qq1][1]
...                         q2[qq1][2] = self.poslist[qq1][2]
...                         q2vel[qq1][0] = self.vellist[qq1][0]
...                         q2vel[qq1][1] = self.vellist[qq1][1]
...                         q2vel[qq1][2] = self.vellist[qq1][2]
...                     else:
...                         q2.append([self.poslist[qq1][0], self.poslist[qq1][1], self.poslist[qq1][2]])
...                         q2vel.append([self.vellist[qq1][0], self.vellist[qq1][1], self.vellist[qq1][2]])
...                 elif (self.poslist[qq1][0] < self.xmid) & (self.poslist[qq1][1] < self.ymid) & (self.poslist[qq1][2] < self.zmid):
...                     if qq1 == 0:
...                         q3[qq1][0] = self.poslist[qq1][0]
...                         q3[qq1][1] = self.poslist[qq1][1]
...                         q3[qq1][2] = self.poslist[qq1][2]
...                         q3vel[qq1][0] = self.vellist[qq1][0]
...                         q3vel[qq1][1] = self.vellist[qq1][1]
...                         q3vel[qq1][2] = self.vellist[qq1][2]
...                     else:
...                         q3.append([self.poslist[qq1][0], self.poslist[qq1][1], self.poslist[qq1][2]])
...                         q3vel.append([self.vellist[qq1][0], self.vellist[qq1][1], self.vellist[qq1][2]])
...                 elif (self.poslist[qq1][0] > self.xmid) & (self.poslist[qq1][1] < self.ymid) & (self.poslist[qq1][2] < self.zmid):
...                     if qq1 == 0:
...                         q4[qq1][0] = self.poslist[qq1][0]
...                         q4[qq1][1] = self.poslist[qq1][1]
...                         q4[qq1][2] = self.poslist[qq1][2]
...                         q4vel[qq1][0] = self.vellist[qq1][0]
...                         q4vel[qq1][1] = self.vellist[qq1][1]
...                         q4vel[qq1][2] = self.vellist[qq1][2]
...                     else:
...                         q4.append([self.poslist[qq1][0], self.poslist[qq1][1], self.poslist[qq1][2]])
...                         q4vel.append([self.vellist[qq1][0], self.vellist[qq1][1], self.vellist[qq1][2]])
...                 elif (self.poslist[qq1][0] < self.xmid) & (self.poslist[qq1][1] > self.ymid) & (self.poslist[qq1][2] > self.zmid):
...                     if qq1 == 0:
...                         q5[qq1][0] = self.poslist[qq1][0]
...                         q5[qq1][1] = self.poslist[qq1][1]
...                         q5[qq1][2] = self.poslist[qq1][2]
...                         q5vel[qq1][0] = self.vellist[qq1][0]
...                         q5vel[qq1][1] = self.vellist[qq1][1]
...                         q5vel[qq1][2] = self.vellist[qq1][2]
...                     else:
...                         q5.append([self.poslist[qq1][0], self.poslist[qq1][1], self.poslist[qq1][2]])
...                         q5vel.append([self.vellist[qq1][0], self.vellist[qq1][1], self.vellist[qq1][2]])
...                 elif (self.poslist[qq1][0] > self.xmid) & (self.poslist[qq1][1] > self.ymid) & (self.poslist[qq1][2] > self.zmid):
...                     if qq1 == 0:
...                         q6[qq1][0] = self.poslist[qq1][0]
...                         q6[qq1][1] = self.poslist[qq1][1]
...                         q6[qq1][2] = self.poslist[qq1][2]
...                         q6vel[qq1][0] = self.vellist[qq1][0]
...                         q6vel[qq1][1] = self.vellist[qq1][1]
...                         q6vel[qq1][2] = self.vellist[qq1][2]
...                     else:
...                         q6.append([self.poslist[qq1][0], self.poslist[qq1][1], self.poslist[qq1][2]])
...                         q6vel.append([self.vellist[qq1][0], self.vellist[qq1][1], self.vellist[qq1][2]])
...                 elif (self.poslist[qq1][0] < self.xmid) & (self.poslist[qq1][1] < self.ymid) & (self.poslist[qq1][2] > self.zmid):
...                     if qq1 == 0:
...                         q7[qq1][0] = self.poslist[qq1][0]
...                         q7[qq1][1] = self.poslist[qq1][1]
...                         q7[qq1][2] = self.poslist[qq1][2]
...                         q7vel[qq1][0] = self.vellist[qq1][0]
...                         q7vel[qq1][1] = self.vellist[qq1][1]
...                         q7vel[qq1][2] = self.vellist[qq1][2]
...                     else:
...                         q7.append([self.poslist[qq1][0], self.poslist[qq1][1], self.poslist[qq1][2]])
...                         q7vel.append([self.vellist[qq1][0], self.vellist[qq1][1], self.vellist[qq1][2]])
...                 elif (self.poslist[qq1][0] > self.xmid) & (self.poslist[qq1][1] < self.ymid) & (self.poslist[qq1][2] > self.zmid):
...                     if qq1 == 0:
...                         q8[qq1][0] = self.poslist[qq1][0]
...                         q8[qq1][1] = self.poslist[qq1][1]
...                         q8[qq1][2] = self.poslist[qq1][2]
...                         q8vel[qq1][0] = self.vellist[qq1][0]
...                         q8vel[qq1][1] = self.vellist[qq1][1]
...                         q8vel[qq1][2] = self.vellist[qq1][2]
...                     else:
...                         q8.append([self.poslist[qq1][0], self.poslist[qq1][1], self.poslist[qq1][2]])
...                         q8vel.append([self.vellist[qq1][0], self.vellist[qq1][1], self.vellist[qq1][2]])
...
...         if len(q1) > 1:
...             self.children.append(OctoTree(q1, q1vel, self.xmin, self.ymid, self.zmin, self.xmid, self.ymax, self.zmid, self.L, self.dt, \
...                                               self.G, self.depth-1))
...
...         if len(q2) > 1:
...             self.children.append(OctoTree(q2, q2vel, self.xmid, self.ymid, self.zmin, self.xmax, self.ymax, self.zmid, self.L, self.dt, \
...                                               self.G, self.depth-1))
...
...         if len(q3) > 1:
...             self.children.append(OctoTree(q3, q3vel, self.xmin, self.ymin, self.zmin, self.xmid, self.ymid, self.zmid, self.L, self.dt, \
...                                               self.G, self.depth-1))
...
...         if len(q4) > 1:
...             self.children.append(OctoTree(q4, q4vel, self.xmid, self.ymin, self.zmin, self.xmax, self.ymid, self.zmid, self.L, self.dt, \
...                                               self.G, self.depth-1))
...         if len(q5) > 1:
...             self.children.append(OctoTree(q5, q5vel, self.xmin, self.ymid, self.zmid, self.xmid, self.ymax, self.zmax, self.L, self.dt, \
...                                               self.G, self.depth-1))
...
...         if len(q6) > 1:
...             self.children.append(OctoTree(q6, q6vel, self.xmid, self.ymid, self.zmid, self.xmax, self.ymax, self.zmax, self.L, self.dt, \
...                                               self.G, self.depth-1))
...
...         if len(q7) > 1:
...             self.children.append(OctoTree(q7, q7vel, self.xmin, self.ymin, self.zmid, self.xmid, self.ymid, self.zmax, self.L, self.dt, \
...                                               self.G, self.depth-1))
...
...         if len(q8) > 1:
...             self.children.append(OctoTree(q8, q8vel, self.xmid, self.ymin, self.zmid, self.xmax, self.ymid, self.zmax, self.L, self.dt, \
...                                               self.G, self.depth-1))
...
...     def CalcF(self, list particle):
...         cdef int ii, jj, ll
...         cdef double CMr, F1x, F1y, F1z, sumposx, sumposy
...         cdef double sumposz
...         cdef double x
...         cdef double y
...         cdef double z
...         cdef double m
...         cdef double CM[3]
...         cdef double CMrvec[3]
...         cdef double CMrsq
...         global F1x, F1y, F1z
...
...         if self.sizesx == self.L:
...             F1x = 0
...             F1y = 0
...             F1z = 0
...         if len(self.poslist) != 0:
...             x = particle[0]
...             y = particle[1]
...             z = particle[2]
...             m = particle[3]
...             sumposx = 0
...             sumposy = 0
...             sumposz = 0
...             summass = 0
...             for ii in range(len(self.poslist)):
...                 sumposx += self.poslist[ii][3] * self.poslist[ii][0]
...                 sumposy += self.poslist[ii][3] * self.poslist[ii][1]
...                 sumposz += self.poslist[ii][3] * self.poslist[ii][2]
...                 summass += self.poslist[ii][3]
...             CM = [sumposx / summass, sumposy / summass, sumposz / summass]
...             CMrvec[0] = CM[0] - x + 0.01
...             CMrvec[1] = CM[1] - y + 0.01
...             CMrvec[2] = CM[2] - z + 0.01
...             CMrsq = CMrvec[0] * CMrvec[0] + CMrvec[1] * CMrvec[1] + CMrvec[2] * CMrvec[2]
...             CMr = CMrsq**(0.5)
...             if (self.sizesx /CMr < 1) or (self.children==[]):
...                 F1x += - (self.G * m * summass /(CMr*CMrsq)) * CMrvec[0]
...                 F1y += - (self.G * m * summass /(CMr*CMrsq)) * CMrvec[1]
...                 F1z += - (self.G * m * summass /(CMr*CMrsq)) * CMrvec[2]
...             else:
...                 for jj in range(len(self.children)):
...                     self.children[jj].CalcF(particle)
...         return F1x, F1y, F1z
...
...
...     def CalcTF(self):
...         cdef double F[256][3]
...         cdef int j
...
...         for j in range(len(self.poslist)):
...             F[j] = self.CalcF(self.poslist[j])
...         self.F = F
...         return F
...
...     def MoveParticles(self):
...         cdef int k, k2
...
...         for k in range(len(self.poslist)):
...             # Calculate velocity, 1st step
...             self.vellist[k][0] += 0.5 * self.F[k][0] * self.dt
...             self.vellist[k][1] += 0.5 * self.F[k][1] * self.dt
...             self.vellist[k][2] += 0.5 * self.F[k][2] * self.dt
...             # Calculate new positions
...             self.poslist[k][0] += self.vellist[k][0] * self.dt
...             self.poslist[k][1] += self.vellist[k][1] * self.dt
...             self.poslist[k][2] += self.vellist[k][2] * self.dt
...             self.poslist[k][0] = self.poslist[k][0] % self.L
...             self.poslist[k][1] = self.poslist[k][1] % self.L
...             self.poslist[k][2] = self.poslist[k][2] % self.L
...
...
...         self.F = self.CalcTF()
...         for k2 in range(len(self.poslist)):
...             # Calculate velocity, 2nd step
...             self.vellist[k2][0] += 0.5 * self.F[k2][0] * self.dt
...             self.vellist[k2][1] += 0.5 * self.F[k2][1] * self.dt
...             self.vellist[k2][2] += 0.5 * self.F[k2][2] * self.dt
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
... ### REMEMBER ###
... # Change the lists izes for F in init and CalcTF when changing N,
... # and recompile.
... N = 256 # Number of particles
...
>>> sys.setrecursionlimit(N)
>>> L = 10000 # [AU]
>>> dt = 1 # [years]
>>> # Generate random positions
... ids = np.linspace(1, N, N)
>>> #randpos = np.random.uniform(0, L, (N, 2))
... randr = np.random.uniform(-L/20, L/20, N)
>>> randphi = np.random.uniform(0, 2*np.pi, N)
>>> randtheta = np.random.uniform(-np.pi, np.pi, N)
>>> randx = randr * np.cos(randphi) * np.sin(randtheta) + L/2
>>> randy = randr * np.sin(randphi) * np.sin(randtheta) + L/2
>>> randz = randr * np.cos(randtheta) + L/2
>>> randveltheta = np.random.normal(0, np.sqrt(100), N)
>>> randvelx = - randveltheta * np.sin(randtheta)
>>> randvely = randveltheta * np.cos(randtheta)
...
>>> mass = np.ones(N) # [Solar mass]
>>> poslist = np.zeros((N, 4))
>>> vellist = np.zeros((N, 3))
>>> #poslist[:, 1:3] = randpos[:,0:]
... poslist[:, 0] = randx
>>> poslist[:, 1] = randy
>>> poslist[:, 2] = randz
>>> poslist[:, 3] = mass
...
>>> #vellist[:, 1:] = np.random.normal(0, np.sqrt(100), (N, 2))#vellist[:, 1:] = np.random.normal(0, np.sqrt(10), (N, 2))
... vellist[:, 0] = randvelx
>>> vellist[:, 1] = randvely
...
>>> #G = 3.9e67 # [AU]**2/([Solar mass] * [year]**2)
... G = 1
...
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111, projection='3d')
>>> ax.scatter(poslist[:,0], poslist[:,1], poslist[:,2], c='r', marker='o', s=poslist[:,3])
...
>>> ax.set_xlabel('X Label')
>>> ax.set_ylabel('Y Label')
>>> ax.set_zlabel('Z Label')
>>> ax.set_zlim(0, L)
>>> ax.set_xlim(0, L)
>>> ax.set_ylim(0, L)
>>> plt.show()
```

```python
>>> %%timeit
... nt = 100
... a = OctoTree(poslist.tolist(), vellist.tolist(), 0, 0, 0, L, L, L, L, dt, G, N)
... a.Simulate(nt)
1 loop, best of 3: 1.87 s per loop
```

```python
>>> xx = np.zeros(N)
>>> yy = np.zeros(N)
>>> zz = np.zeros(N)
>>> for iii in range(len(poslist)):
...     xx[iii] = a.poslist[iii][0]
...     yy[iii] = a.poslist[iii][1]
...     zz[iii] = a.poslist[iii][2]
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111, projection='3d')
>>> ax.scatter(xx, yy, zz, c='r', marker='o', s=poslist[:,3])
...
>>> ax.set_xlabel('X Label')
>>> ax.set_ylabel('Y Label')
>>> ax.set_zlabel('Z Label')
>>> ax.set_zlim(0, L)
>>> ax.set_xlim(0, L)
>>> ax.set_ylim(0, L)
>>> plt.show()
```

```python
>>> # Plotting code
... a = OctoTree(poslist.tolist(), vellist.tolist(), 0, 0, 0, L, L, L, L, dt, G, N)
...
>>> fig = plt.figure()
>>> ax = p3.Axes3D(fig)
>>> particles, = ax.plot(poslist[:,0], poslist[:,1], poslist[:,2], 'bo', ms=6)
...
>>> def animate(i):
...     pposlist = a.MoveParticles()
...     for iii in range(len(poslist)):
...         xx[iii] = pposlist[iii][0]
...         yy[iii] = pposlist[iii][1]
...         zz[iii] = pposlist[iii][2]
...     particles.set_data(xx, yy)
...     particles.set_3d_properties(zz)
...     return particles
...
>>> def save_anim(file, title):
...     "saves the animation with a desired title"
...     Writer = animation.writers['ffmpeg']
...     writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
...     file.save(title + '.mp4', writer=writer)
...
...
>>> ax.set_xlim3d([0, L])
>>> ax.set_ylim3d([0, L])
>>> ax.set_zlim3d([0, L])
>>> ani = animation.FuncAnimation(fig, animate, frames=1000, repeat=False)
>>> #save_anim(ani, 'Test')
... plt.show()
```

```python
>>> a = [[1, 2, 2], [3, 4, 2]]
>>> print(a)
>>> print(len(a))
>>> a[1][1]
```

```python
>>> len(poslist.tolist())
```

```python
>>> print m
```

```python
>>> print(m)
```

```python
>>> m = [1.0]
>>> m.append(2)
>>> print(m)
```

```python

```
