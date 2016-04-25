```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> %matplotlib inline
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
>>> xmax, ymax = np.amax(poslist, axis=0)[1:3]
>>> xmin, ymin = np.amin(poslist, axis=0)[1:3]
...
>>> q1 = poslist[(poslist[:,1] < L/2) & (poslist[:,2] > L/2)]
>>> q2 = poslist[(poslist[:,1] > L/2) & (poslist[:,2] > L/2)]
>>> q3 = poslist[(poslist[:,1] < L/2) & (poslist[:,2] < L/2)]
>>> q4 = poslist[(poslist[:,1] > L/2) & (poslist[:,2] < L/2)]
>>> plt.figure()
>>> plt.scatter(q4[:,1], q4[:,2])
>>> plt.xlim([0,10])
>>> plt.ylim([0,10])
>>> plt.show()
```

```python

```
