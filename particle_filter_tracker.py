import numpy as np
import sympy as sp
import cv2

from ex2_utils import get_patch, create_epanechnik_kernel, extract_histogram, show_image
from utils.tracker import Tracker

def normalize_histogram(histogram):
    bin_sum = sum(histogram)
    return np.array([el / bin_sum for el in histogram])


def sample_gauss(mu, sigma, n):
    # sample n samples from a given multivariate normal distribution
    return np.random.multivariate_normal(mu, sigma, n)


def get_dynamic_model_matrices(q, model):
    if model == "RW":
        F = [[0, 0],
             [0, 0]]

        L = [[1, 0],
             [0, 1]]
    elif model == "NCV":
        F = [[0, 0, 1, 0],
             [0, 0, 0, 1],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]

        L = [[0, 0],
             [0, 0],
             [1, 0],
             [0, 1]]
    elif model == "NCA":
        F = [[0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]]

        L = [[0, 0],
             [0, 0],
             [0, 0],
             [0, 0],
             [1, 0],
             [0, 1]]

    T = sp.symbols('T')
    F = sp.Matrix(F)
    Fi = (sp.exp(F * T)).subs(T, 1)

    L = sp.Matrix(L)
    Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))
    Q = Q.subs(T, 1)

    # F = np.array(F, dtype="float32")
    Fi = np.array(Fi, dtype="float32")
    Q = np.array(Q, dtype="float32")

    return Fi, Q


def hellinger(p, q):
    return 1 / np.sqrt(2) * np.linalg.norm(np.sqrt(p) - np.sqrt(q))



def dist_to_prob(dist, sigma):
    return np.exp(-0.5 * dist ** 2 / sigma ** 2)


def draw_particles(image, particles, weights, position, size, color):
    image2 = image.copy()
    for (x, y, _, _), weight in zip(particles, weights):
        r = np.random.randint(0, 255)
        g = np.random.randint(0, 255)
        b = np.random.randint(0, 255)
        thickness = int(weight / 0.002) - 1
        image2 = cv2.circle(image2, (int(x), int(y)), radius=0, color=(b, g, r), thickness=thickness)
    image2, _ = get_patch(image2, position, (size[0] * 2.5, size[1] * 2.5))
    image2 = cv2.resize(image2, dsize=(int(image2.shape[1] * 3), int(image2.shape[0] * 3)))
    show_image(image2, 0, "-")


def change_colorspace(image, colorspace):
    if colorspace == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if colorspace == "LAB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    if colorspace == "RGB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if colorspace == "YCRCB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)


class PFTracker(Tracker):

    def name(self):
        return "PFTracker"

    def initialize(self, image, region):

        region = [int(el) for el in region]

        if (region[2] % 2 == 0):
            region[2] += 1
        if (region[3] % 2 == 0):
            region[3] += 1

        self.kernel_sigma = 0.5
        self.histogram_bins = 8
        self.n_of_particles = 50
        self.enlarge_factor = 2
        self.distance_sigma = 0.11
        self.update_alpha = 0.01
        self.color_change = (True, "YCRCB")
        self.draw_particles = False
        self.dynamic_model = "NCV"

        if self.color_change[0]:
            image = change_colorspace(image, self.color_change[1])

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.window = max(region[2], region[3]) * self.enlarge_factor
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])
        self.template, _ = get_patch(image, self.position, self.size)

        image_pl = image.shape[0] * image.shape[1]
        patch_pl = self.size[0] * self.size[1]

        q = int(patch_pl / image_pl * 200)
        self.q = q if q > 0 else 1
        #self.q = 100

        # CREATING VISUAL MODEL
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.kernel_sigma)
        self.patch_size = self.kernel.shape
        self.template_histogram = normalize_histogram(extract_histogram(self.template,
                                                                        self.histogram_bins,
                                                                        weights=self.kernel))

        # GENERATING DYNAMIC MODEL MATRICES
        self.system_matrix, self.system_covariance = get_dynamic_model_matrices(self.q, self.dynamic_model)
        self.particle_state = [self.position[0], self.position[1]]

        if self.dynamic_model == "NCV":
            self.particle_state.extend([0, 0])
        if self.dynamic_model == "NCA":
            self.particle_state.extend([0, 0, 0, 0])

        # GENERATING N PARTICLES AROUND POSITION
        self.particles = sample_gauss(self.particle_state, self.system_covariance, self.n_of_particles)

        self.weights = np.array([1 / self.n_of_particles for _ in range(self.n_of_particles)])

    def track(self, image):

        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]]

        if self.color_change[0]:
            image = change_colorspace(image, self.color_change[1])

        # PARTICLE SAMPLING
        weights_cumsumed = np.cumsum(self.weights)
        rand_samples = np.random.rand(self.n_of_particles, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)
        particles_new = self.particles[sampled_idxs.flatten(), :]

        noises = sample_gauss([0 for _ in range(self.system_matrix.shape[0])], self.system_covariance, self.n_of_particles)
        self.particles = np.transpose(np.matmul(self.system_matrix, np.transpose(particles_new))) + noises

        for index, p in enumerate(particles_new):
            p_x = self.particles[index][0]
            p_y = self.particles[index][1]

            try:
                patch, _ = get_patch(image, (p_x, p_y), self.patch_size)
                histogram = normalize_histogram(extract_histogram(patch, self.histogram_bins, weights=self.kernel))
                hell_dist = hellinger(histogram, self.template_histogram)
                prob = dist_to_prob(hell_dist, self.distance_sigma)
            except Exception as e:
                prob = 0

            self.weights[index] = prob

        # NORMALIZE WEIGHTS
        self.weights = self.weights / np.sum(self.weights)

        # DRAWING PARTICLES
        if self.draw_particles:
            draw_particles(image, self.particles, self.weights, self.position, self.size, (255, 0, 0))

        # COMPUTE NEW POSITION
        new_x = sum([particle[0] * self.weights[index] for index, particle in enumerate(self.particles)])
        new_y = sum([particle[1] * self.weights[index] for index, particle in enumerate(self.particles)])

        self.position = (new_x, new_y)

        # UPDATE VISUAL MODEL
        if self.update_alpha > 0:
            self.template, _ = get_patch(image, (new_x, new_y), self.patch_size)
            self.template_histogram = (1 - self.update_alpha) * self.template_histogram + self.update_alpha * normalize_histogram(
                extract_histogram(self.template, self.histogram_bins, weights=self.kernel))

        return [new_x - self.size[0] / 2, new_y - self.size[1] / 2, self.size[0], self.size[1]]
