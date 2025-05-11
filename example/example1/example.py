import numpy as np

a = np.array([i for i in range(10)]) + 10
for k in enumerate(a):
    print(k)


for k in enumerate(a[1:], 1):
    print(k)
