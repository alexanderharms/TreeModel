```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
```

```python
>>> # Variables
... N = 128 # Number of particles
>>> L = 10
...
>>> ids = np.linspace(1, N, N)
>>> randpos = np.random.uniform(0, L, (N, 2))
>>> poslist = np.zeros((N, 3))
>>> poslist[:, 0] = ids
>>> poslist[:, 1:] = randpos[:,0:]
...
>>> # Divide
... # First block 0 is the entire volume.
... blocks[0]
>>> blocks = np.zeros((4, 5))
>>> blocks
```

```python
>>> plt.figure()
>>> plt.scatter(poslist[:,1],poslist[:,2])
>>> plt.show()
```

```python

```
