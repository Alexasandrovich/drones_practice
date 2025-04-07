#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class LidarProcessor:
    def __init__(self):
        """Инициализация параметров и ROS-топиков."""
        self.theta_step = math.pi / 180  # Шаг по углу (1 градус)
        self.theta_bins = 180  # Количество угловых бинов
        self.rho_resolution = 0.05  # Разрешение по rho (метры)

        # Паблишер для визуализации стен и углов
        self.marker_pub = rospy.Publisher("/walls_markers", MarkerArray, queue_size=1)

        # Подписка на /scan
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)

    def scan_callback(self, msg):
        """Обработка данных лидара."""
        # Преобразование данных в координаты (x, y)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)
        valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max)
        points = np.array([
            (dist * math.cos(ang), dist * math.sin(ang))
            for dist, ang, valid in zip(ranges, angles, valid_mask) if valid
        ])
        if len(points) == 0:
            return

        # Поиск линий с помощью Хаффа
        lines = self.find_lines_hough(points)

        # Нахождение углов
        corners = self.find_corners(lines)
        print(corners)

        # Публикация маркеров
        self.publish_markers(lines, corners, msg.header)

    def find_lines_hough(self, points):
        """Поиск четырёх линий, образующих прямоугольник, с помощью преобразования Хаффа."""
        max_rho = np.max(np.hypot(points[:, 0], points[:, 1]))
        rho_bins = int(2 * max_rho / self.rho_resolution) + 1
        accumulator = np.zeros((rho_bins, self.theta_bins))

        # Заполнение аккумулятора
        for x, y in points:
            for theta_idx in range(self.theta_bins):
                theta = theta_idx * self.theta_step
                rho = x * math.cos(theta) + y * math.sin(theta)
                rho_idx = int((rho + max_rho) / self.rho_resolution)
                if 0 <= rho_idx < rho_bins:
                    accumulator[rho_idx, theta_idx] += 1

        # Сумма голосов по rho для каждого theta
        sum_over_rho = np.sum(accumulator, axis=0)  # shape (theta_bins,)

        # Нахождение двух перпендикулярных направлений
        total_sums = []
        for theta_idx in range(self.theta_bins):
            theta_idx2 = (theta_idx + 90) % self.theta_bins
            total_sum = sum_over_rho[theta_idx] + sum_over_rho[theta_idx2]
            total_sums.append(total_sum)
        theta_idx_max = np.argmax(total_sums)
        theta1_idx = theta_idx_max
        theta2_idx = (theta_idx_max + 90) % self.theta_bins

        # Для theta1: два rho с наибольшими голосами
        rho_indices1 = np.argsort(accumulator[:, theta1_idx])[-2:][::-1]
        rho_values1 = [-max_rho + rho_idx * self.rho_resolution for rho_idx in rho_indices1]

        # Для theta2: два rho с наибольшими голосами
        rho_indices2 = np.argsort(accumulator[:, theta2_idx])[-2:][::-1]
        rho_values2 = [-max_rho + rho_idx * self.rho_resolution for rho_idx in rho_indices2]

        # Формирование четырёх линий
        lines = [
            (rho_values1[0], theta1_idx * self.theta_step),
            (rho_values1[1], theta1_idx * self.theta_step),
            (rho_values2[0], theta2_idx * self.theta_step),
            (rho_values2[1], theta2_idx * self.theta_step),
        ]

        return lines

    def find_corners(self, lines):
        """Нахождение углов как пересечений линий."""
        corners = []
        # Пересечения линий с theta1 (0, 1) и theta2 (2, 3)
        for i in [0, 1]:
            for j in [2, 3]:
                rho1, theta1 = lines[i]
                rho2, theta2 = lines[j]
                A = np.array([[math.cos(theta1), math.sin(theta1)],
                              [math.cos(theta2), math.sin(theta2)]])
                b = np.array([rho1, rho2])
                try:
                    x, y = np.linalg.solve(A, b)
                    if math.hypot(x, y) < 10:  # Ограничение по дальности
                        corners.append((x, y))
                except np.linalg.LinAlgError:
                    pass  # Линии параллельны
        return corners

    def publish_markers(self, lines, corners, header):
        """Публикация маркеров стен и углов в RViz."""
        ma = MarkerArray()
        mid = 0
        for rho, theta in lines:
            P = np.array([rho * math.cos(theta), rho * math.sin(theta)])
            V = np.array([-math.sin(theta), math.cos(theta)])
            start = P - 10 * V  # Длинные линии для визуализации
            end = P + 10 * V
            mk = Marker()
            mk.header = header
            mk.id = mid
            mk.type = Marker.LINE_STRIP
            mk.scale.x = 0.03
            mk.color.b = 1.0
            mk.color.r = 1.0
            mk.color.a = 1.0
            mk.points = [Point(start[0], start[1], 0), Point(end[0], end[1], 0)]
            ma.markers.append(mk)
            mid += 1

        if corners:
            mk_points = Marker()
            mk_points.header = header
            mk_points.id = mid
            mk_points.type = Marker.POINTS
            mk_points.scale.x = 0.1
            mk_points.scale.y = 0.1
            mk_points.color.r = 1.0
            mk_points.color.b = 1.0
            mk_points.color.a = 1.0
            mk_points.points = [Point(x, y, 0) for x, y in corners]
            ma.markers.append(mk_points)

        self.marker_pub.publish(ma)

if __name__ == "__main__":
    rospy.init_node("lidar_processor")
    processor = LidarProcessor()
    rospy.spin()