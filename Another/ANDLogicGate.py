import numpy as np
n = 0.1;
t = np.array([-1, -1, -1, 1])
x = np.array([[1, 1, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1]])
w = np.array([0.1, 0.1, 0.1])
for epoch in range (0, 10):
    for i in range (0, 4):
        o = np.matmul(w, x[:,i])
        o = np.where(o >= 0,1, -1)
        dw = n*(t[i]-o)*x[:,i]
        w = w+dw
print(np.sign(np.matmul((w),x)))
