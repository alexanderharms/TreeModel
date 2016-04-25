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
>>> q1 = Split4(poslist, 0, 0, L, L)[0]
>>> plt.figure()
>>> plt.scatter(q1[:,1],q1[:,2])
>>> plt.xlim([0,10])
>>> plt.ylim([0,10])
>>> plt.show()
```

```python
>>> q11 = Split4(q1, 0, L/2, L/2, L)[0]
>>> plt.figure()
>>> plt.scatter(q11[:,1],q11[:,2])
>>> plt.xlim([0,10])
>>> plt.ylim([0,10])
>>> plt.show()
```

```python

```

```python

```
