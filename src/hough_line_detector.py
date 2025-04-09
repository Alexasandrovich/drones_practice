import numpy as np
from base_line_detector import LineDetector
import math

# Детектор линий с использованием преобразования Хаффа
class HoughLineDetector(LineDetector):
    def __init__(self, theta_step=math.pi / 180, theta_bins=180, rho_resolution=0.05):
        self.theta_step = theta_step
        self.theta_bins = theta_bins
        self.rho_resolution = rho_resolution

    def detect_lines(self, points):
        max_rho = np.max(np.hypot(points[:, 0], points[:, 1]))
        rho_bins = int(2 * max_rho / self.rho_resolution) + 1
        accumulator = np.zeros((rho_bins, self.theta_bins))

        for x, y in points:
            for theta_idx in range(self.theta_bins):
                theta = theta_idx * self.theta_step
                rho = x * math.cos(theta) + y * math.sin(theta)
                rho_idx = int((rho + max_rho) / self.rho_resolution)
                if 0 <= rho_idx < rho_bins:
                    accumulator[rho_idx, theta_idx] += 1

        sum_over_rho = np.sum(accumulator, axis=0)
        total_sums = []
        for theta_idx in range(self.theta_bins):
            theta_idx2 = (theta_idx + 90) % self.theta_bins
            total_sum = sum_over_rho[theta_idx] + sum_over_rho[theta_idx2]
            total_sums.append(total_sum)
        theta_idx_max = np.argmax(total_sums)
        theta1_idx = theta_idx_max
        theta2_idx = (theta_idx_max + 90) % self.theta_bins

        rho_indices1 = np.argsort(accumulator[:, theta1_idx])[-2:][::-1]
        rho_values1 = [-max_rho + rho_idx * self.rho_resolution for rho_idx in rho_indices1]
        rho_indices2 = np.argsort(accumulator[:, theta2_idx])[-2:][::-1]
        rho_values2 = [-max_rho + rho_idx * self.rho_resolution for rho_idx in rho_indices2]

        lines = [
            (rho_values1[0], theta1_idx * self.theta_step),
            (rho_values1[1], theta1_idx * self.theta_step),
            (rho_values2[0], theta2_idx * self.theta_step),
            (rho_values2[1], theta2_idx * self.theta_step),
        ]
        return lines