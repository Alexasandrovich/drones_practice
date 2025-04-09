#!/usr/bin/env python
import rospy
import math
from clover import srv
from std_srvs.srv import Trigger

rospy.init_node('lab1_flight')

# Инициализация сервисов
get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
land = rospy.ServiceProxy('land', Trigger)


def safe_takeoff(height=1.5, speed=0.5):
    # Взлет с проверкой достижения высоты
    navigate(z=height, speed=speed, frame_id='body', auto_arm=True)
    rospy.sleep(2)
    while get_telemetry().z < height - 0.1:
        rospy.sleep(0.2)


def move_to_point(x, y, z, speed=0.4, timeout=5):
    # Движение к точке с таймаутом
    start = rospy.get_time()
    navigate(x=x, y=y, z=z, speed=speed, frame_id='aruco_map')
    while not rospy.is_shutdown():
        if rospy.get_time() - start > timeout:
            break
        rospy.sleep(0.2)


if __name__ == "__main__":
    safe_takeoff(1.5)

    # Квадратная траектория
    points = [
        (1, 0, 1.5),
        (1, 1, 1.5),
        (0, 1, 1.5),
        (0, 0, 1.5)
    ]

    for x, y, z in points:
        move_to_point(x, y, z)

    land()