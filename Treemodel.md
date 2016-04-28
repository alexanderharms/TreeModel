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
...     cdef public sizesx, sizesy, sizesz, children, xmid, ymid, zmid, Fx, Fy, Fz
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
...         cdef double Fx[3]
...         cdef double Fy[3]
...         cdef double Fz[3]
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
...         self.Fx, self.Fy, self.Fz = self.CalcTF()
...
...     def CreateTree(self):
...
...         cdef double q1[1][3]
...         cdef double q2[1][3]
...         cdef double q3[1][3]
...         cdef double q4[1][3]
...         cdef double q5[1][3]
...         cdef double q6[1][3]
...         cdef double q7[1][3]
...         cdef double q8[1][3]
...         cdef double q1vel[1][3]
...         cdef double q2vel[1][3]
...         cdef double q3vel[1][3]
...         cdef double q4vel[1][3]
...         cdef double q5vel[1][3]
...         cdef double q6vel[1][3]
...         cdef double q7vel[1][3]
...         cdef double q8vel[1][3]
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
...             self.children.append(OctoTree(q1, q1vel, self.xmin, self.ymid, self.zmin, self.xmid, self.ymax, self.zmid, self.L/2, self.dt, \
...                                               self.G, self.depth-1))
...
...         if len(q2) > 1:
...             self.children.append(OctoTree(q2, q2vel, self.xmid, self.ymid, self.zmin, self.xmax, self.ymax, self.zmid, self.L/2, self.dt, \
...                                               self.G, self.depth-1))
...
...         if len(q3) > 1:
...             self.children.append(OctoTree(q3, q3vel, self.xmin, self.ymin, self.zmin, self.xmid, self.ymid, self.zmid, self.L/2, self.dt, \
...                                               self.G, self.depth-1))
...
...         if len(q4) > 1:
...             self.children.append(OctoTree(q4, q4vel, self.xmid, self.ymin, self.zmin, self.xmax, self.ymid, self.zmid, self.L/2, self.dt, \
...                                               self.G, self.depth-1))
...         if len(q5) > 1:
...             self.children.append(OctoTree(q5, q5vel, self.xmin, self.ymid, self.zmid, self.xmid, self.ymax, self.zmax, self.L/2, self.dt, \
...                                               self.G, self.depth-1))
...
...         if len(q6) > 1:
...             self.children.append(OctoTree(q6, q6vel, self.xmid, self.ymid, self.zmid, self.xmax, self.ymax, self.zmax, self.L/2, self.dt, \
...                                               self.G, self.depth-1))
...
...         if len(q7) > 1:
...             self.children.append(OctoTree(q7, q7vel, self.xmin, self.ymin, self.zmid, self.xmid, self.ymid, self.zmax, self.L/2, self.dt, \
...                                               self.G, self.depth-1))
...
...         if len(q8) > 1:
...             self.children.append(OctoTree(q8, q8vel, self.xmid, self.ymin, self.zmid, self.xmax, self.ymid, self.zmax, self.L/2, self.dt, \
...                                               self.G, self.depth-1))
...
...     def CalcF(self, list particle, double F1x, double F1y, double F1z):
...         cdef int ii, jj
...         cdef double CMr, sumposx, sumposy
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
...             CMrvec[0] = CM[0] - x
...             CMrvec[1] = CM[1] - y
...             CMrvec[2] = CM[2] - z
...             CMrsq = CMrvec[0] * CMrvec[0] + CMrvec[1] * CMrvec[1] + CMrvec[2] * CMrvec[2]
...             CMr = CMrsq**(0.5)
...             if (self.sizesx / CMr < 0.5) or (self.children==[]):
...                 F1x += (self.G * m * summass / (CMrsq + 0.1)) * (CMrvec[0]/CMr)
...                 F1y += (self.G * m * summass / (CMrsq + 0.1)) * (CMrvec[1]/CMr)
...                 F1z += (self.G * m * summass / (CMrsq + 0.1)) * (CMrvec[2]/CMr)
...             else:
...                 for jj in range(len(self.children)):
...                     self.children[jj].CalcF(particle, F1x, F1y, F1z)
...         return F1x, F1y, F1z
...
...
...     def CalcTF(self):
...         cdef double Fx[3]
...         cdef double Fy[3]
...         cdef double Fz[3]
...         cdef int j, jj
...         for jj in range(len(Fx)):
...             Fx[jj] = 0
...             Fy[jj] = 0
...             Fz[jj] = 0
...
...         for j in range(len(self.poslist)):
...             Fx[j], Fy[j], Fz[j] = self.CalcF(self.poslist[j], Fx[j], Fy[j], Fz[j])
...
...         self.Fx = Fx
...         self.Fy = Fy
...         self.Fz = Fz
...         return Fx, Fy, Fz
...
...     def MoveParticles(self):
...         cdef int k, k2
...
...         for k in range(len(self.poslist)):
...             # Calculate velocity, 1st step
...             self.vellist[k][0] += 0.5 * self.Fx[k] * self.dt
...             self.vellist[k][1] += 0.5 * self.Fy[k] * self.dt
...             self.vellist[k][2] += 0.5 * self.Fz[k] * self.dt
...             # Calculate new positions
...             self.poslist[k][0] += self.vellist[k][0] * self.dt
...             self.poslist[k][1] += self.vellist[k][1] * self.dt
...             self.poslist[k][2] += self.vellist[k][2] * self.dt
...             self.poslist[k][0] = self.poslist[k][0] % 10
...             self.poslist[k][1] = self.poslist[k][1] % 10
...             self.poslist[k][2] = self.poslist[k][2] % 10
...
...
...         self.Fx, self.Fy, self.Fz = self.CalcTF()
...         for k2 in range(len(self.poslist)):
...             # Calculate velocity, 2nd step
...             self.vellist[k2][0] += 0.5 * self.Fx[k2] * self.dt
...             self.vellist[k2][1] += 0.5 * self.Fy[k2] * self.dt
...             self.vellist[k2][2] += 0.5 * self.Fz[k2] * self.dt
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
... # Change the list sizes for F in init and CalcTF when changing N,
... # and recompile.
... N = 3 # Number of particles
...
>>> #sys.setrecursionlimit(2*N)
... L = 10 # [AU]
>>> dt = 0.01 # [years]
>>> a = 0.5
>>> G = 1
>>> M = N
>>> # Generate random positions
...
... #randpos = np.random.uniform(0, L, (N, 2))
... #randr = np.random.uniform(-L/20, L/20, N)
... # Pr = np.random.uniform(0, 1, N)
... # r = a * np.sqrt(Pr) / (np.sqrt(Pr) - 1)
... # mass = M / (2 * np.pi) * (a/r) * (1 / (r + a ) ** 3)
... #mass = (Mh * r**2) / (r + a)**2
... # Erand = np.random.uniform(-2, 0, N)
... # fErand = np.random.uniform(0, 0.3, N)
...
... # def hernquist(Erand):
... #     q = np.sqrt(- a * Erand / (G * M))
... #     fE = (np.sqrt(M)/(8*np.pi**3 * a ** (5/2) * np.sqrt(2*G)))  \
... #     * (1/(1-q**2)**(5/2)) * (3*np.arcsin(q) + (q * (1 - q**2)**(1/2)) * (1 - 2*q**2) \
... #     * (8 * q**4 - 8 * q**2 - 3))
... #     return fE
...
... # def potential(r):
... #     potential = G * M / (np.absolute(r) + a)
... #     return potential
...
... # fE = hernquist(Erand)
... # fEcheck = fE > fErand
... # fEcheck.astype(int)
... # E = Erand[np.nonzero(Erand * fEcheck)]
... # sumfEcheck = sum(fEcheck)
... # while sumfEcheck < N:
... #     Erand2 = np.random.uniform(-2, 0, (N - sum(fEcheck)))
... #     fErand2 = np.random.uniform(0, 0.3, (N - sum(fEcheck)))
... #     fE2 = hernquist(Erand2)
... #     fEcheck2 = fE2 > fErand2
... #     fEcheck2.astype(int)
... #     E2 = Erand2[np.nonzero(Erand2 * fEcheck2)]
... #     E = np.append(E, E2)
... #     sumfEcheck += sum(fEcheck2)
...
... # velmagsq = np.absolute(2*(E[0:N] - potential(r)))
...
... # randvelx = np.random.uniform(-np.sqrt(velmagsq), np.sqrt(velmagsq), N)
... # randvely = np.random.uniform(-np.sqrt(np.absolute(velmagsq - randvelx**2)), \
... #                             np.sqrt(np.absolute(velmagsq - randvelx**2)), N)
... # randvelz = np.random.uniform(-np.sqrt(np.absolute(velmagsq - randvelx**2 - randvely**2)), \
... #                             np.sqrt(np.absolute(velmagsq - randvelx**2 - randvely**2)))
... # r = np.random.uniform(-L/2, L/2, N)
... # randphi = np.random.uniform(0, 2*np.pi, N)
... # randtheta = np.random.uniform(-np.pi, np.pi, N)
... theta = [0, 2*np.pi/3, 4*np.pi/3]
>>> r = np.sqrt(0.5**2+1**2)*0.5
>>> v = np.sqrt(3)
>>> mass = np.ones(N) # [Solar mass]
>>> poslist = np.zeros((N, 4))
>>> vellist = np.zeros((N, 3))
>>> #poslist[:, 1:3] = randpos[:,0:]
... poslist[0, 0] = r + L/2
>>> poslist[0, 1] = L/2
>>> poslist[0, 2] = L/2
>>> poslist[0, 3] = 1
>>> poslist[1, 0] = r*np.cos(theta[1]) + L/2
>>> poslist[1, 1] = r*np.sin(theta[1]) + L/2
>>> poslist[1, 2] = L/2
>>> poslist[1, 3] = 1
>>> poslist[2, 0] = r*np.cos(theta[2]) + L/2
>>> poslist[2, 1] = r*np.sin(theta[2]) + L/2
>>> poslist[2, 2] = L/2
>>> poslist[2, 3] = 1
>>> #vellist[:, 1:] = np.random.normal(0, np.sqrt(100), (N, 2))#vellist[:, 1:] = np.random.normal(0, np.sqrt(10), (N, 2))
... # vellist[:, 0] = randvelx
... # vellist[:, 1] = randvely
... # vellist[:, 2] = randvely
... vellist[0, 0] = 0
>>> vellist[0, 1] = v
>>> vellist[0, 2] = 0
>>> vellist[1, 0] = v*-np.sin(theta[1])
>>> vellist[1, 1] = v*np.cos(theta[1])
>>> vellist[1, 2] = 0
>>> vellist[2, 0] = v*-np.sin(theta[2])
>>> vellist[2, 1] = v*np.cos(theta[2])
>>> vellist[2, 2] = 0
>>> # vellist[2, :] = [np.sin(np.pi/3), np.cos(np.pi/3), 0]
...
... #G = 3.9e67 # [AU]**2/([Solar mass] * [year]**2)
...
... fig = plt.figure()
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
>>> print(poslist)
[[ 5.55901699  5.          5.          1.        ]
 [ 4.7204915   5.48412292  5.          1.        ]
 [ 4.7204915   4.51587708  5.          1.        ]]
```

```python
>>> # Plotting code
... aa = OctoTree(poslist.tolist(), vellist.tolist(), 0, 0, 0, L, L, L, L, dt, G, 2*N)
...
>>> fig = plt.figure()
>>> ax = p3.Axes3D(fig)
>>> particles, = ax.plot(poslist[:,0], poslist[:,1], poslist[:,2], 'bo', ms=6)
...
>>> xx = np.zeros(N)
>>> yy = np.zeros(N)
>>> zz = np.zeros(N)
...
>>> def animate(i):
...     pposlist = aa.MoveParticles()
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
>>> ax.set_zlim(-L/2, L/2)
>>> ax.set_xlim(-L/2, L/2)
>>> ax.set_ylim(-L/2, L/2)
>>> plt.show()
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
>>> a = OctoTree(poslist.tolist(), vellist.tolist(), 0, 0, 0, L, L, L, L, dt, G, N)
>>> print(a.children)
```

```python
>>> print(aa.CalcF(poslist[0,:].tolist(),0, 0, 0))
(0.003024460473401977, -0.000795844256372009, 0.007143460815929137)
```

```python
>>> np.sqrt(0.5**2+1)*0.5
0.55901699437494745
```

```python

```
