```python
>>> #%matplotlib inline
... %load_ext Cython
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from matplotlib import animation
>>> import mpl_toolkits.mplot3d.axes3d as p3
>>> import sys
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
...         cdef double Fx[4]
...         cdef double Fy[4]
...         cdef double Fz[4]
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
...                 if (self.poslist[qq1][0] <= self.xmid) & (self.poslist[qq1][1] > self.ymid) & (self.poslist[qq1][2] <= self.zmid):
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
...                 elif (self.poslist[qq1][0] > self.xmid) & (self.poslist[qq1][1] > self.ymid) & (self.poslist[qq1][2] <= self.zmid):
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
...                 elif (self.poslist[qq1][0] <= self.xmid) & (self.poslist[qq1][1] <= self.ymid) & (self.poslist[qq1][2] <= self.zmid):
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
...                 elif (self.poslist[qq1][0] > self.xmid) & (self.poslist[qq1][1] <= self.ymid) & (self.poslist[qq1][2] <= self.zmid):
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
...                 elif (self.poslist[qq1][0] <= self.xmid) & (self.poslist[qq1][1] > self.ymid) & (self.poslist[qq1][2] > self.zmid):
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
...                 elif (self.poslist[qq1][0] <= self.xmid) & (self.poslist[qq1][1] <= self.ymid) & (self.poslist[qq1][2] > self.zmid):
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
...                 elif (self.poslist[qq1][0] > self.xmid) & (self.poslist[qq1][1] <= self.ymid) & (self.poslist[qq1][2] > self.zmid):
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
...                 F1x += (self.G * m * summass / (CMrsq + 0.2)) * (CMrvec[0]/CMr)
...                 F1y += (self.G * m * summass / (CMrsq + 0.2)) * (CMrvec[1]/CMr)
...                 F1z += (self.G * m * summass / (CMrsq + 0.2)) * (CMrvec[2]/CMr)
...             else:
...                 for jj in range(len(self.children)):
...                     self.children[jj].CalcF(particle, F1x, F1y, F1z)
...         return F1x, F1y, F1z
...
...
...     def CalcTF(self):
...         cdef double Fx[4]
...         cdef double Fy[4]
...         cdef double Fz[4]
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
...             self.poslist[k][0] = self.poslist[k][0] % 20
...             self.poslist[k][1] = self.poslist[k][1] % 20
...             self.poslist[k][2] = self.poslist[k][2] % 20
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
...
... L = 20 # [AU]
>>> dt = 0.1 # [years]
>>> G = 1
...
>>> r = np.linspace(5,5,1)
>>> c = 1
>>> Nr = [np.floor(c*r)]
>>> theta = np.pi/2
>>> phi1 = 2*np.pi*(1-1/Nr[0])
>>> phi = np.linspace(0, phi1 , Nr[0])
>>> print(phi)
>>> print(0, 2*np.pi/3, 4*np.pi/3)
>>> poslist = np.zeros((len(phi), 4))
>>> vellist = np.zeros((len(phi), 3))
>>> for j in range(len(phi)):
...     poslist[j, 0] = r[0] * np.sin(theta) * np.cos(phi[j])
...     poslist[j, 1] = r[0] * np.sin(theta) * np.sin(phi[j])
...     poslist[j, 2] = r[0] * np.cos(theta)
...     poslist[j, 3] = 1
>>> mass = np.sum(poslist[:,3])
>>> print(mass)
>>> vmag = np.sqrt(G*mass/r[0])
>>> for k in range(len(phi)):
...     vellist[k, 0] = -vmag * np.sin(phi[k])
...     vellist[k, 1] = vmag * np.cos(phi[k])
...     vellist[k, 2] = 0
...
>>> for i in range(1,len(Nr)):
...     phi1 = 2*np.pi*(1-1/Nr[i])
...     phi = np.linspace(0, phi1, Nr[i])
...     phi = np.delete(phi, -1)
...     for jj in range(len(phi)):
...         poslist2 = np.zeros(4)
...         poslist2[0] = r[i] * np.sin(theta) * np.cos(phi[jj])
...         poslist2[1] = r[i] * np.sin(theta) * np.sin(phi[jj])
...         poslist2[2] = r[i] * np.cos(theta)
...         poslist2[3] = 1
...         poslist = np.vstack((poslist, poslist2))
...     mass = np.sum(poslist[:,3])
...     vmag = np.sqrt(G*mass/r[0])
...     for kk in range(len(phi)):
...         vellist2 = np.zeros(3)
...         vellist2[0] = -vmag * np.sin(phi[kk])
...         vellist2[1] = vmag * np.cos(phi[kk])
...         vellist2[2] = 0
...         vellist = np.vstack((vellist, vellist2))
...
>>> poslist[:,0:4] = poslist[:,0:4] + L/2
>>> totalmass = np.sum(poslist[:,3])
>>> N = np.int(np.sum(Nr)) # Number of particles
>>> print(N)
>>> #sys.setrecursionlimit(2*N)
... ## Three particles rotating
... # r = np.sqrt(2**2 + 4**2)*0.5
... # theta = [0 , 2*np.pi/3, 4*np.pi/3]
... # totalmass = 3
... # poslist = np.zeros((4,4))
... # vellist = np.zeros((4,3))
... # totalmass = 4
... # theta = [0, np.pi/2, np.pi, 3*np.pi/2]
... # r = np.sqrt(2**2 + 2**2)
... # v = np.sqrt(G*totalmass/r)
... # poslist[0, 0] = r + L/2
... # poslist[0, 1] = L/2
... # poslist[0, 2] = L/2
... # poslist[0, 3] = 1
... # poslist[1, 0] = r*np.cos(theta[1]) + L/2
... # poslist[1, 1] = r*np.sin(theta[1]) + L/2
... # poslist[1, 2] = L/2
... # poslist[1, 3] = 1
... # poslist[2, 0] = r*np.cos(theta[2]) + L/2
... # poslist[2, 1] = r*np.sin(theta[2]) + L/2
... # poslist[2, 2] = L/2
... # poslist[2, 3] = 1
... # poslist[3, 0] = r*np.cos(theta[3]) + L/2
... # poslist[3, 1] = r*np.sin(theta[3]) + L/2
... # poslist[3, 2] = L/2
... # poslist[3, 3] = 1
... # vellist[0, 0] = 0
... # vellist[0, 1] = v
... # vellist[0, 2] = 0
... # vellist[1, 0] = v*-np.sin(theta[1])
... # vellist[1, 1] = v*np.cos(theta[1])
... # vellist[1, 2] = 0
... # vellist[2, 0] = v*-np.sin(theta[2])
... # vellist[2, 1] = v*np.cos(theta[2])
... # vellist[2, 2] = 0
... # vellist[3, 0] = v*-np.sin(theta[3])
... # vellist[3, 1] = v*np.cos(theta[3])
... # vellist[3, 2] = 0
... # vellist[2, :] = [np.sin(np.pi/3), np.cos(np.pi/3), 0]
...
... fig = plt.figure()
>>> ax = fig.add_subplot(111, projection='3d')
>>> ax.scatter(poslist[:,0], poslist[:,1], poslist[:,2], c='r', marker='o')
...
>>> ax.set_xlabel('X Label')
>>> ax.set_ylabel('Y Label')
>>> ax.set_zlabel('Z Label')
>>> ax.set_zlim(0, L)
>>> ax.set_xlim(0, L)
>>> ax.set_ylim(0, L)
>>> plt.show()
>>> print(poslist)
[ 0.          1.57079633  3.14159265  4.71238898]
0 2.0943951023931953 4.1887902047863905
4.0
4
[[ 12.82842712  10.          10.           1.        ]
 [ 10.          12.82842712  10.           1.        ]
 [  7.17157288  10.          10.           1.        ]
 [ 10.           7.17157288  10.           1.        ]]
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
...     for iii in range(len(pposlist)):
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
10.118920711500271
10.237636339587048
10.355942234026292
10.473634609413924
10.59051097377904
10.706370552642479
10.821014707064736
10.934247344258214
11.045875319389413
11.155708827255884
11.263561782590259
11.36925218781851
11.472602487180998
11.573439906211997
11.671596775665295
11.76691083906918
11.859225543192778
11.94839031080618
12.034260795218337
12.116699116178175
12.195574076825075
12.270761361473816
12.342143714115569
12.409611097609757
12.473060833631067
12.532397723520758
12.587534150271466
12.638390161949317
12.684893536926012
12.726979831356504
12.7645924093946
12.797682456689259
12.826208977748479
12.850138777795477
12.86944642977344
12.88411422718062
12.894132123437174
12.899497658499042
12.900215873442743
12.896299213748385
12.887767422006856
12.874647420771407
12.856973186263982
12.834785613633118
12.808132374443305
12.777067767055827
12.741652560538613
12.701953832717832
12.658044802957297
12.610004660223435
12.55791838696403
12.501876579298388
12.44197526398531
12.378315712603566
12.311004253347686
12.240152080809946
12.165875064087826
12.088293553524945
12.007532186362772
11.923719691550492
11.83698869393128
11.747475517995074
11.655319991360892
11.560665248125822
11.463657532193086
11.364446000668224
11.263182527390386
11.160021506645048
11.055119657085262
10.948635825870767
10.840730793018011
10.73156707593935
10.621308734136417
```

```python
>>> nt = 10
>>> aa = OctoTree(poslist.tolist(), vellist.tolist(), 0, 0, 0, L, L, L, L, dt, G, N)
>>> aa.Simulate(nt)
>>> xx = np.zeros(N)
>>> yy = np.zeros(N)
>>> zz = np.zeros(N)
>>> for iii in range(len(aa.poslist)):
...     xx[iii] = aa.poslist[iii][0]
...     yy[iii] = aa.poslist[iii][1]
...     zz[iii] = aa.poslist[iii][2]
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111, projection='3d')
>>> ax.scatter(xx, yy, zz, c='r', marker='o')
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
>>> cols = np.linspace(0,5,6)
>>> print(cols)
[ 0.  1.  2.  3.  4.  5.]
```

```python
>>> np.sqrt(0.5**2+1)*0.5
0.55901699437494745
```

```python

```
