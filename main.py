from ex4_utils import kalman_step, compute_input_matrices, get_model_matrices
import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(42)


def generate_trajectory(N):
    v = np.array([np.random.randint(0, 100) for _ in range(N)])
    x = np.cos(v) * v
    y = np.sin(v) * v
    return x, y


q = 1
r = 1
MODEL = "NCV"
N_OF_TRAJECTORIES = 3

F, L, H, R = get_model_matrices(MODEL, r)
Fi, Q = compute_input_matrices(q, F, L)


N = 20
# v = np.linspace(5 * math.pi, 0, N)
# x = np.cos(v) * v
# y = np.sin(v) * v

for _ in range(N_OF_TRAJECTORIES):
    x, y = generate_trajectory(N)

    # plt.scatter(x, y, edgecolors="red")
    plt.plot(x, y, c="blue", linewidth=1)

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

    plt.plot(sx, sy, c="red", linewidth=1)
    plt.show()
    plt.clf()
