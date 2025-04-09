import numpy as np
from base_line_detector import LineDetector
import math

# Детектор линий с использованием RANSAC
class RansacLineDetector(LineDetector):
    def __init__(self, num_iterations=100, threshold=0.1, min_inliers=50):
        self.num_iterations = num_iterations
        self.threshold = threshold
        self.min_inliers = min_inliers

    def detect_lines(self, points, num_lines=4):
        lines = []
        remaining_points = points.copy()
        for _ in range(num_lines):
            best_line = None
            best_inliers = []
            for _ in range(self.num_iterations):
                idx1, idx2 = np.random.choice(len(remaining_points), 2, replace=False)
                p1, p2 = remaining_points[idx1], remaining_points[idx2]
                line = self.points_to_line(p1, p2)
                inliers = self.get_inliers(line, remaining_points)
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_line = line
            if len(best_inliers) >= self.min_inliers:
                lines.append(best_line)
                remaining_points = np.delete(remaining_points, best_inliers, axis=0)
            else:
                break
        return lines

    def points_to_line(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        theta = math.atan2(dy, dx)
        rho = x1 * math.cos(theta) + y1 * math.sin(theta)
        return rho, theta

    def get_inliers(self, line, points):
        rho, theta = line
        inliers = []
        for i, (x, y) in enumerate(points):
            distance = abs(x * math.cos(theta) + y * math.sin(theta) - rho)
            if distance < self.threshold:
                inliers.append(i)
        return inliers