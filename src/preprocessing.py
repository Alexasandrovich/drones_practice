import numpy as np

# Класс для фильтрации шума скана, так как без него детекции краёв будут плохо работать
class ScanPreprocesor:
    def compute_convex_hull(self, points):
        """Вычисляет выпуклую оболочку точек с использованием алгоритма Грэхема."""
        if len(points) < 3:
            return points  # Если точек меньше 3, оболочка не строится

        # Находим точку с минимальной y-координатой (при равенстве — минимальной x)
        start_idx = np.argmin(points[:, 1])
        start_point = points[start_idx]
        hull_points = [start_point]
        points = np.delete(points, start_idx, axis=0)

        # Вычисляем полярные углы относительно начальной точки
        vectors = points - start_point
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]

        # Строим выпуклую оболочку
        for point in sorted_points:
            while (len(hull_points) >= 2 and
                   np.cross(hull_points[-1] - hull_points[-2], point - hull_points[-2]) <= 0):
                hull_points.pop()
            hull_points.append(point)

        return np.array(hull_points)


    def point_to_line_distance(self, point, line_start, line_end):
        """Вычисляет расстояние от точки до отрезка."""
        p = np.array(point)
        a = np.array(line_start)
        b = np.array(line_end)
        ab = b - a
        ap = p - a
        bp = p - b

        # Если проекция точки лежит вне отрезка
        if np.dot(ab, ap) < 0:
            return np.hypot(ap[0], ap[1])  # Расстояние до a
        elif np.dot(-ab, bp) < 0:
            return np.hypot(bp[0], bp[1])  # Расстояние до b
        # Если проекция внутри отрезка
        else:
            return abs(np.cross(ab, ap)) / np.hypot(ab[0], ab[1])  # Перпендикулярное расстояние


    def preprocess_points(self, points, distance_threshold=0.5):
        """Фильтрует точки, оставляя те, что находятся не дальше distance_threshold от выпуклой оболочки."""
        # Вычисляем выпуклую壳очку
        hull_points = self.compute_convex_hull(points)
        if len(hull_points) < 3:
            return points  # Если оболочка не построена, возвращаем исходные точки

        # Фильтруем точки
        filtered_points = []
        for point in points:
            # Находим минимальное расстояние до всех рёбер оболочки
            min_distance = min(
                self.point_to_line_distance(point, hull_points[i], hull_points[(i + 1) % len(hull_points)])
                for i in range(len(hull_points))
            )
            if min_distance <= distance_threshold:
                filtered_points.append(point)

        return np.array(filtered_points)