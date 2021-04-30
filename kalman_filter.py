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


def plot_curves(q, r, x, y, model, ax, title):
    F, L, H, R = get_model_matrices(model, r)
    Fi, Q = compute_input_matrices(q, F, L)

    ax.plot(x, y, c="red", linewidth=1)

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

    ax.plot(sx, sy, c="blue", linewidth=1)
    ax.title.set_text(title)


N = 40
v = np.linspace(5 * math.pi, 0, N)
x = np.cos(v) * v
y = np.sin(v) * v

curves = []
curves.append((x,y))
curves.append((
    np.array([4, 6, 7, 7, 6, 4, 3, 3, 5, 7, 8, 10, 12, 13, 13]),
    np.array([-4, -3, -1, 1, 3, 4, 6, 8, 9, 8, 6, 5, 6, 8, 10])
))

curves.append((
    np.array([1, 3, 4, 4, 4, 3, 1, -1, -3, -5, -6, -6, -6, -5, -3, -1]),
    np.array([2, 2, 0, -2, -4, -6, -7, -7, -7, -6, -4, -2, 0, 2, 2, 2])
))

for index, (x, y) in enumerate(curves):
    # fig1, ((ax1_11, ax1_12, ax1_13, ax1_14, ax1_15),
    #        (ax1_21, ax1_22, ax1_23, ax1_24, ax1_25),
    #        (ax1_31, ax1_32, ax1_33, ax1_34, ax1_35)) = plt.subplots(3, 5, figsize=(15, 10))

    fig1, ((ax1_11, ax1_12, ax1_14, ax1_15),
           (ax1_21, ax1_22, ax1_24, ax1_25),
           (ax1_31, ax1_32, ax1_34, ax1_35)) = plt.subplots(3, 4, figsize=(12, 9))

    model = "RW"
    plot_curves(100, 1, x, y, model, ax1_11, model + ": q = 100, r = 1")
    plot_curves(5, 1, x, y, model, ax1_12, model + ": q = 5, r = 1")
    # plot_curves(1, 1, x, y, model, ax1_13, model + ": q = 1, r = 1")
    plot_curves(1, 5, x, y, model, ax1_14, model + ": q = 1, r = 5")
    plot_curves(1, 100, x, y, model, ax1_15, model + ": q = 1, r = 100")

    model = "NCV"
    plot_curves(100, 1, x, y, model, ax1_21, model + ": q = 100, r = 1")
    plot_curves(5, 1, x, y, model, ax1_22, model + ": q = 5, r = 1")
    # plot_curves(1, 1, x, y, model, ax1_23, model + ": q = 1, r = 1")
    plot_curves(1, 5, x, y, model, ax1_24, model + ": q = 1, r = 5")
    plot_curves(1, 100, x, y, model, ax1_25, model + ": q = 1, r = 100")

    model = "NCA"
    plot_curves(100, 1, x, y, model, ax1_31, model + ": q = 100, r = 1")
    plot_curves(5, 1, x, y, model, ax1_32, model + ": q = 5, r = 1")
    # plot_curves(1, 1, x, y, model, ax1_33, model + ": q = 1, r = 1")
    plot_curves(1, 5, x, y, model, ax1_34, model + ": q = 1, r = 5")
    plot_curves(1, 100, x, y, model, ax1_35, model + ": q = 1, r = 100")

    plt.savefig("curve" + str(index) + ".png")
    plt.show()