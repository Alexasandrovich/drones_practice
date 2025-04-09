import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from hough_line_detector import HoughLineDetector
from ransac_line_detector import RansacLineDetector
from preprocessing import ScanPreprocesor

# Основной класс обработки данных лидара
class LidarProcessor:
    def __init__(self, map_corners, method="ransac"):
        self.map_corners = map_corners
        self.preprocessor = ScanPreprocesor()
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

        points = self.preprocessor.preprocess_points(points, distance_threshold=0.5)

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
    method = "hough"  # или "ransac"
    processor = LidarProcessor(map_corners, method=method)
    rospy.spin()