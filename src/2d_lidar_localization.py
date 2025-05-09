import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from hough_line_detector import HoughLineDetector
from ransac_line_detector import RansacLineDetector
from preprocessing import ScanPreprocesor

# Основной класс обработки данных лидара
class LidarProcessor:
    def __init__(self, map_corners, method="ransac"):
        self.map_corners = map_corners
        self.current_pose = [0, 0, 0]  # x, y, yaw
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

        # Подписка на данные лидара
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)

        # Публикация GT карты один раз при инициализации
        self.center_x = sum(c[0] for c in map_corners) / 4
        self.center_y = sum(c[1] for c in map_corners) / 4
        self.shifted_map_corners = [(x - self.center_x, y - self.center_y) for x, y in map_corners]

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
        robot_pose = self.match_corners(detected_corners)

        # Публикация калибровочного скана в GT системе координат
        self.publish_markers(points, lines, detected_corners, robot_pose, msg.header)

    def get_current_pose(self):
        return self.current_pose


    def publish_gt_map(self, robot_pose):
        """Публикует статичную GT карту в топике /gt_map с центром в (0, 0)"""
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
            p1 = Point(self.shifted_map_corners[i][0], self.shifted_map_corners[i][1], 0)
            p2 = Point(self.shifted_map_corners[(i + 1) % 4][0], self.shifted_map_corners[(i + 1) % 4][1], 0)
            gt_marker.points.append(p1)
            gt_marker.points.append(p2)
        ma.markers.append(gt_marker)

        # Публикация ориентации дрона в виде стрелки
        rospy.logwarn(f"robot pose = {robot_pose}")
        x, y, yaw = robot_pose
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "map"
        arrow_marker.ns = "robot_arrow"
        arrow_marker.id = 3
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        arrow_marker.scale.x = 0.1  # длина стрелки
        arrow_marker.scale.y = 0.1  # толщина хвоста
        arrow_marker.scale.z = 0.1  # толщина головы
        arrow_marker.color.r = 0.0
        arrow_marker.color.g = 1.0
        arrow_marker.color.b = 0.0
        arrow_marker.color.a = 1.0

        start_point = Point(x, y, 0)
        end_point = Point(x + math.cos(yaw) * 0.5, y + math.sin(yaw) * 0.5, 0)  # вектор ориентации

        arrow_marker.points.append(start_point)
        arrow_marker.points.append(end_point)

        ma.markers.append(arrow_marker)
        self.gt_map_pub.publish(ma)

    def rho_theta_to_points(self, rho, theta, length=10):
        """Преобразует параметры прямой в две точки для отрисовки"""
        x0 = rho * math.cos(theta)
        y0 = rho * math.sin(theta)
        dx = -math.sin(theta) * length / 2
        dy = math.cos(theta) * length / 2
        p1 = (x0 + dx, y0 + dy)
        p2 = (x0 - dx, y0 - dy)
        return p1, p2

    def publish_markers(self, points, lines, detected_corners, robot_pose, header):
        marker_array = MarkerArray()

        # Публикация исходного скана
        calibration_marker = Marker()
        calibration_marker.header.frame_id = "map"
        calibration_marker.ns = "calibration_scan"
        calibration_marker.id = 0
        calibration_marker.type = Marker.POINTS
        calibration_marker.action = Marker.ADD
        calibration_marker.scale.x = 0.05
        calibration_marker.scale.y = 0.05
        calibration_marker.color.r = 1.0
        calibration_marker.color.a = 1.0
        calibration_marker.points = [Point(x, y, 0) for x, y in points]
        marker_array.markers.append(calibration_marker)

        # Публикация линий
        line_marker = Marker()
        line_marker.header.frame_id = "map"
        line_marker.ns = "calibration_lines"
        line_marker.id = 1
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.03
        line_marker.color.g = 1.0
        line_marker.color.a = 1.0
        for rho, theta in lines:
            p1, p2 = self.rho_theta_to_points(rho, theta)
            line_marker.points.append(Point(p1[0], p1[1], 0))
            line_marker.points.append(Point(p2[0], p2[1], 0))
        marker_array.markers.append(line_marker)

        # Публикация углов
        corner_marker = Marker()
        corner_marker.header.frame_id = "map"
        corner_marker.ns = "calibration_corners"
        corner_marker.id = 2
        corner_marker.type = Marker.SPHERE_LIST
        corner_marker.action = Marker.ADD
        corner_marker.scale.x = 0.08
        corner_marker.scale.y = 0.08
        corner_marker.scale.z = 0.08
        corner_marker.color.b = 1.0
        corner_marker.color.a = 1.0
        corner_marker.points = [Point(x, y, 0) for x, y in detected_corners]
        marker_array.markers.append(corner_marker)

        # Публикация всего массива
        self.marker_pub.publish(marker_array)
        self.publish_gt_map(robot_pose)

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

    def match_corners(self, detected_corners):
        x, y, yaw = 0, 0, 0
        # todo: используйте self.map_corners для нахождения соответствия между углами карты и найденными углами

        self.current_pose = [x, y, yaw]
        return self.current_pose


if __name__ == "__main__":
    rospy.init_node("lidar_processor")
    # обход карты делаем по часовой стрелке из (0, 0)
    map_corners = [(0, 0), (7, 0), (7, 14), (0, 14)]  # Пример: комната 14x7 метра
    method = "hough"  # или "ransac"
    processor = LidarProcessor(map_corners, method=method)
    rospy.spin()