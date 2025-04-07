#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

def scan_callback(msg):
    # 1. Преобразование данных лазера в координаты (x, y)
    angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
    ranges = np.array(msg.ranges)
    valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max)
    points = []
    for dist, ang, valid in zip(ranges, angles, valid_mask):
        if valid:
            x = dist * math.cos(ang)
            y = dist * math.sin(ang)
            points.append((x, y))
    points = np.array(points)
    if len(points) == 0:
        return

    # 2. Преобразование Хаффа для поиска линий
    theta_step = math.pi / 180  # шаг 1 градус
    theta_bins = 180
    rho_resolution = 0.05  # разрешение по rho (метры)
    max_rho = np.max(np.hypot(points[:, 0], points[:, 1]))
    rho_bins = int(2 * max_rho / rho_resolution) + 1
    accumulator = np.zeros((rho_bins, theta_bins))
    for x, y in points:
        for theta_index in range(theta_bins):
            theta = theta_index * theta_step
            rho = x * math.cos(theta) + y * math.sin(theta)
            rho_index = int((rho + max_rho) / rho_resolution)
            if 0 <= rho_index < rho_bins:
                accumulator[rho_index, theta_index] += 1

    # Находим пики (линии)
    N = 10  # максимальное количество линий
    flat_accumulator = accumulator.flatten()
    indices = np.argpartition(flat_accumulator, -N)[-N:]
    peaks = []
    min_votes = 10  # минимальное количество голосов
    for index in indices:
        if flat_accumulator[index] >= min_votes:
            rho_index = index // theta_bins
            theta_index = index % theta_bins
            rho = -max_rho + rho_index * rho_resolution
            theta = theta_index * theta_step
            peaks.append((rho, theta))

    # 3. Объединение коллинеарных линий
    merged_lines = []
    theta_eps = math.pi / 36  # 5 градусов
    rho_eps = 0.2  # 0.2 метра
    peaks.sort(key=lambda x: x[1])  # сортировка по theta
    i = 0
    while i < len(peaks):
        rho1, theta1 = peaks[i]
        inliers = []
        for j in range(i, len(peaks)):
            rho2, theta2 = peaks[j]
            if abs(theta1 - theta2) < theta_eps and abs(rho1 - rho2) < rho_eps:
                inliers.append(peaks[j])
            else:
                break
        if len(inliers) > 1:
            rhos = [p[0] for p in inliers]
            thetas = [p[1] for p in inliers]
            merged_lines.append((np.mean(rhos), np.mean(thetas)))
            i += len(inliers)
        else:
            merged_lines.append(inliers[0])
            i += 1

    # 4. Выбор четырех линий для прямоугольника
    theta_groups = {}
    for rho, theta in merged_lines:
        theta_mod = theta % math.pi  # приводим к [0, pi)
        found = False
        for group_theta in theta_groups:
            if abs(theta_mod - group_theta) < theta_eps or abs(theta_mod - group_theta - math.pi) < theta_eps:
                theta_groups[group_theta].append((rho, theta))
                found = True
                break
        if not found:
            theta_groups[theta_mod] = [(rho, theta)]

    # Берем две основные группы
    sorted_groups = sorted(theta_groups.items(), key=lambda x: len(x[1]), reverse=True)
    if len(sorted_groups) < 2:
        rospy.logwarn("Недостаточно линий для прямоугольника")
        return
    group1 = sorted_groups[0][1]
    group2 = sorted_groups[1][1]

    # Выбираем лучшую линию из группы по количеству inliers
    def get_best_line(group):
        best_line, max_inliers = None, 0
        for rho, theta in group:
            distances = np.abs(points[:, 0] * math.cos(theta) + points[:, 1] * math.sin(theta) - rho)
            inliers_count = np.sum(distances < 0.05)
            if inliers_count > max_inliers:
                max_inliers = inliers_count
                best_line = (rho, theta)
        return best_line

    line1 = get_best_line(group1)
    line2 = get_best_line(group2)
    theta1, theta2 = line1[1], line2[1]
    line3 = max(merged_lines, key=lambda l: np.sum(np.abs(points[:, 0] * math.cos(l[1]) + points[:, 1] * math.sin(l[1]) - l[0]) < 0.05) if abs((l[1] % math.pi) - (theta1 + math.pi / 2) % math.pi) < theta_eps else 0)
    line4 = max(merged_lines, key=lambda l: np.sum(np.abs(points[:, 0] * math.cos(l[1]) + points[:, 1] * math.sin(l[1]) - l[0]) < 0.05) if abs((l[1] % math.pi) - (theta2 + math.pi / 2) % math.pi) < theta_eps else 0)
    selected_lines = [line1, line2, line3, line4]

    # 5. Определение отрезков стен
    wall_segments = []
    for rho, theta in selected_lines:
        distances = np.abs(points[:, 0] * math.cos(theta) + points[:, 1] * math.sin(theta) - rho)
        inliers = points[distances < 0.05]
        if len(inliers) < 2:
            continue
        P = np.array([rho * math.cos(theta), rho * math.sin(theta)])
        V = np.array([-math.sin(theta), math.cos(theta)])
        t_values = np.dot(inliers - P, V)
        start_point = P + np.min(t_values) * V
        end_point = P + np.max(t_values) * V
        wall_segments.append((start_point, end_point))

    # 6. Нахождение углов (пересечений)
    intersections = []
    for i in range(4):
        for j in range(i + 1, 4):
            rho1, theta1 = selected_lines[i]
            rho2, theta2 = selected_lines[j]
            A, B, C = math.cos(theta1), math.sin(theta1), rho1
            D, E, F = math.cos(theta2), math.sin(theta2), rho2
            denom = A * E - B * D
            if abs(denom) > 1e-6:
                x = (C * E - B * F) / denom
                y = (A * F - C * D) / denom
                if math.hypot(x, y) < msg.range_max:
                    intersections.append((x, y))

    # 7. Визуализация в RViz
    ma = MarkerArray()
    mid = 0
    for start, end in wall_segments:  # Отрезки стен (синий)
        mk = Marker()
        mk.header = msg.header
        mk.id = mid
        mid += 1
        mk.type = Marker.LINE_STRIP
        mk.scale.x = 0.03
        mk.color.b = 1.0
        mk.color.r = 1.0
        mk.color.a = 1.0
        mk.points.append(Point(start[0], start[1], 0))
        mk.points.append(Point(end[0], end[1], 0))
        ma.markers.append(mk)

    if intersections:  # Углы (красный)
        mk_points = Marker()
        mk_points.header = msg.header
        mk_points.id = mid
        mk_points.type = Marker.POINTS
        mk_points.scale.x = 0.1
        mk_points.scale.y = 0.1
        mk_points.color.r = 1.0
        mk_points.color.b = 1.0
        mk_points.color.a = 1.0
        for x, y in intersections:
            mk_points.points.append(Point(x, y, 0))
        ma.markers.append(mk_points)

    pub.publish(ma)

# Инициализация ROS
rospy.init_node("walls_detector")
pub = rospy.Publisher("/walls_markers", MarkerArray, queue_size=1)
rospy.Subscriber("/scan", LaserScan, scan_callback)
rospy.spin()