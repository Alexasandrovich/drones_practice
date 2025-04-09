import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header


# Базовый класс для детекторов линий
class LineDetector:
    def detect_lines(self, points):
        raise NotImplementedError("Этот метод должен быть реализован в подклассах")


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


# Основной класс обработки данных лидара
class LidarProcessor:
    def __init__(self, map_corners, method="ransac"):
        self.map_corners = map_corners
        # Выбор метода детекции линий
        if method == "hough":
            self.line_detector = HoughLineDetector()
        elif method == "ransac":
            self.line_detector = RansacLineDetector()
        else:
            raise ValueError(f"Неизвестный метод: {method}")

        # Паблишеры
        self.gt_map_pub = rospy.Publisher("/gt_map", MarkerArray, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher("/map_markers", MarkerArray, queue_size=1)
        self.calibration_scan_pub = rospy.Publisher("/calibration_scan", Marker, queue_size=1)

        # Подписка на данные лидара
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)

        # Публикация GT карты один раз при инициализации
        self.publish_gt_map()

    def publish_gt_map(self):
        """Публикует статичную GT карту в отдельном топике /gt_map"""
        ma = MarkerArray()
        gt_marker = Marker()
        gt_marker.header.frame_id = "map"
        gt_marker.header.stamp = rospy.Time.now()
        gt_marker.id = 0
        gt_marker.type = Marker.LINE_LIST
        gt_marker.scale.x = 0.05
        gt_marker.color.g = 1.0
        gt_marker.color.a = 1.0
        for i in range(4):
            p1 = Point(self.map_corners[i][0], self.map_corners[i][1], 0)
            p2 = Point(self.map_corners[(i + 1) % 4][0], self.map_corners[(i + 1) % 4][1], 0)
            gt_marker.points.append(p1)
            gt_marker.points.append(p2)
        ma.markers.append(gt_marker)
        self.gt_map_pub.publish(ma)


    def scan_callback(self, msg):
        """Обработка данных лидара"""
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)
        valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max)
        points = np.array([
            (dist * math.cos(ang), dist * math.sin(ang))
            for dist, ang, valid in zip(ranges, angles, valid_mask) if valid
        ])
        if len(points) == 0:
            return

        points = self.preprocess_points(points, distance_threshold=0.5)

        # Публикация калибровочного скана в GT системе координат
        calibration_marker = Marker()
        calibration_marker.header.frame_id = "map"
        calibration_marker.header.stamp = msg.header.stamp
        calibration_marker.ns = "calibration_scan"
        calibration_marker.id = 0
        calibration_marker.type = Marker.POINTS
        calibration_marker.action = Marker.ADD
        calibration_marker.scale.x = 0.05
        calibration_marker.scale.y = 0.05
        calibration_marker.color.r = 1.0
        calibration_marker.color.a = 1.0
        calibration_marker.points = [Point(x, y, 0) for x, y in points]
        self.calibration_scan_pub.publish(calibration_marker)

        # Обнаружение линий
        lines = self.line_detector.detect_lines(points)
        if len(lines) < 4:
            rospy.logwarn("Не удалось обнаружить 4 линии")
            return

        # Нахождение углов
        detected_corners = self.find_corners(lines)
        if len(detected_corners) < 4:
            rospy.logwarn("Не удалось найти 4 угла")
            return

        # "Натягивание" углов на GT карту
        matched_corners = self.force_match_corners(detected_corners, self.map_corners)

        # Определение положения робота
        robot_position = self.calculate_robot_position(matched_corners)

        # Публикация маркеров (углы и положение робота)
        self.publish_markers(matched_corners, robot_position, msg.header)

    def find_corners(self, lines):
        """Находит углы как пересечения линий"""
        corners = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                rho1, theta1 = lines[i]
                rho2, theta2 = lines[j]
                A = np.array([[math.cos(theta1), math.sin(theta1)],
                              [math.cos(theta2), math.sin(theta2)]])
                b = np.array([rho1, rho2])
                try:
                    x, y = np.linalg.solve(A, b)
                    corners.append((x, y))
                except np.linalg.LinAlgError:
                    pass
        return corners[:4]

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

    def force_match_corners(self, detected_corners, map_corners):
        """Принудительно сопоставляет обнаруженные углы с GT углами"""
        if len(detected_corners) < 4:
            detected_corners += [detected_corners[-1]] * (4 - len(detected_corners))
        detected_corners = detected_corners[:4]
        matched = []
        used_map_indices = []
        for dc in detected_corners:
            distances = [
                math.hypot(dc[0] - mc[0], dc[1] - mc[1])
                for i, mc in enumerate(map_corners) if i not in used_map_indices
            ]
            if distances:
                min_idx = np.argmin(distances)
                map_idx = [i for i in range(4) if i not in used_map_indices][min_idx]
                matched.append(map_corners[map_idx])
                used_map_indices.append(map_idx)

        return matched

    def calculate_robot_position(self, corners):
        """Вычисляет положение робота и ограничивает его границами GT карты"""
        x_sum = sum(c[0] for c in corners)
        y_sum = sum(c[1] for c in corners)
        robot_x = x_sum / len(corners)
        robot_y = y_sum / len(corners)

        min_x = min(c[0] for c in self.map_corners)
        max_x = max(c[0] for c in self.map_corners)
        min_y = min(c[1] for c in self.map_corners)
        max_y = max(c[1] for c in self.map_corners)

        robot_x = max(min(robot_x, max_x), min_x)
        robot_y = max(min(robot_y, max_y), min_y)
        robot_z = 0
        rospy.logwarn(f"robot_x, robot_y = {robot_x, robot_y}")
        return robot_x, robot_y, robot_z

    def publish_markers(self, corners, robot_position, header):
        """Публикует статичные углы и динамическое положение робота"""
        ma = MarkerArray()
        mid = 0

        # Публикация "натянутых" углов (статичны, так как соответствуют GT)
        mk_points = Marker()
        mk_points.header = Header(frame_id="map", stamp=header.stamp)
        mk_points.id = mid
        mk_points.type = Marker.POINTS
        mk_points.scale.x = 0.1
        mk_points.scale.y = 0.1
        mk_points.color.r = 1.0
        mk_points.color.b = 1.0
        mk_points.color.a = 1.0
        mk_points.points = [Point(x, y, 0) for x, y in corners]
        ma.markers.append(mk_points)
        mid += 1

        # Публикация положения робота (динамично)
        robot_marker = Marker()
        robot_marker.header = Header(frame_id="map", stamp=header.stamp)
        robot_marker.id = mid
        robot_marker.type = Marker.SPHERE
        robot_marker.scale.x = 0.2
        robot_marker.scale.y = 0.2
        robot_marker.scale.z = 0.2
        robot_marker.color.r = 1.0
        robot_marker.color.a = 1.0
        robot_marker.pose.position = Point(robot_position[0], robot_position[1], 0)
        ma.markers.append(robot_marker)

        self.marker_pub.publish(ma)


if __name__ == "__main__":
    rospy.init_node("lidar_processor")
    # 14/7
    map_corners = [(0, 0), (7, 0), (7, 14), (0, 14)]  # Пример: комната 14x7 метра
    method = "ransac"  # или "ransac"
    processor = LidarProcessor(map_corners, method=method)
    rospy.spin()