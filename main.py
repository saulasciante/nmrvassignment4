from ex4_utils import kalman_step, compute_input_matrices, get_model_matrices
import numpy as np
import math

import matplotlib.pyplot as plt

q = 100
r = 1
model = "NCV"

F, L, H, R = get_model_matrices(model, r)
Fi, Q = compute_input_matrices(q, F, L)


N = 40
v = np.linspace(5 * math.pi, 0, N)
x = np.cos(v) * v
y = np.sin(v) * v

# plt.scatter(x, y, edgecolors="red")
plt.plot(x, y, "blue")

sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

sx[0] = x[0]
sy[0] = y[0]

state = np.zeros((Fi.shape[0], 1), dtype=np.float32).flatten()
state[0] = x[0]
state[1] = y[0]
covariance = np.eye(Fi.shape[0], dtype=np.float32)

for j in range(1, x.size):
    state, covariance, _, _ = kalman_step(Fi, H, Q, R, np.reshape(np.array([x[j] , y [j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
    sx[j] = state[0]
    sy[j] = state[1]

plt.plot(sx, sy, "red")
plt.show()
