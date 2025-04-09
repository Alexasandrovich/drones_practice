#!/usr/bin/env python
import rospy
import math
from clover import srv
from std_srvs.srv import Trigger
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from lidar_processor import LidarProcessor

# Инициализация узла ROS
rospy.init_node('2d_lidar_navigation')

# Сервисы дрона
get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
land = rospy.ServiceProxy('land', Trigger)

# Параметры
WALL_DISTANCE = 0.5  # Целевое расстояние до стены, м
SAFETY_THRESHOLD = 1.0  # Порог ошибки локализации, м
ROOM_CORNERS = [(0, 0), (7, 0), (7, 14), (0, 14)]  # Пример комнаты 14x7 м

# PID параметры для следования вдоль стены
KP = 1.0
KI = 0.0
KD = 0.0
pid_error_sum = 0
pid_last_error = 0


def safe_takeoff(height=1.5, speed=0.5):
    """Взлет с проверкой достижения высоты"""
    navigate(z=height, speed=speed, frame_id='body', auto_arm=True)
    rospy.sleep(2)
    while get_telemetry().z < height - 0.1:
        rospy.sleep(0.2)


def get_distance_to_wall(scan):
    """Определение расстояния до ближайшей стены спереди"""
    front_ranges = scan.ranges[len(scan.ranges) // 2 - 10: len(scan.ranges) // 2 + 10]
    return min(front_ranges)


def pid_controller(error, dt):
    """PID-регулятор для корректировки курса"""
    global pid_error_sum, pid_last_error
    pid_error_sum += error * dt
    derivative = (error - pid_last_error) / dt
    pid_last_error = error
    return KP * error + KI * pid_error_sum + KD * derivative


def follow_wall(target_distance, speed=0.4):
    """Следование вдоль стены с заданным расстоянием"""
    rate = rospy.Rate(10)  # 10 Гц
    while not rospy.is_shutdown():
        scan = rospy.wait_for_message('/scan', LaserScan)
        distance = get_distance_to_wall(scan)
        error = distance - target_distance
        dt = 0.1  # 10 Гц
        correction = pid_controller(error, dt)

        # Корректировка курса
        yaw_rate = correction
        navigate(x=0, y=0, z=0, yaw=math.nan, yaw_rate=yaw_rate, speed=speed, frame_id='body')

        # Проверка на угол
        if distance > target_distance + 0.5:
            return  # Выход для поворота
        rate.sleep()


def turn_90_degrees():
    """Поворот на 90° влево"""
    navigate(yaw=math.radians(90), speed=0.5, frame_id='body')
    rospy.sleep(2)


def check_localization_quality(lidar_pose, telemetry_pose):
    """Проверка качества локализации и аварийная посадка"""
    dx = lidar_pose[0] - telemetry_pose.position.x
    dy = lidar_pose[1] - telemetry_pose.position.y
    error = math.sqrt(dx ** 2 + dy ** 2)
    if error > SAFETY_THRESHOLD:
        rospy.logwarn("Ошибка локализации превышена! Посадка...")
        land()
        rospy.signal_shutdown("Аварийная посадка")


if __name__ == "__main__":
    # Инициализация LidarProcessor
    processor = LidarProcessor(map_corners=ROOM_CORNERS, method="hough")

    # Взлет
    safe_takeoff(1.5)

    # Облет периметра (4 стены)
    for _ in range(4):
        follow_wall(WALL_DISTANCE)
        turn_90_degrees()

        # Проверка локализации
        lidar_pose = processor.get_current_pose()
        telemetry = get_telemetry()
        check_localization_quality(lidar_pose, telemetry)

    # Посадка в центре комнаты
    center_x = sum(c[0] for c in ROOM_CORNERS) / 4
    center_y = sum(c[1] for c in ROOM_CORNERS) / 4
    navigate(x=center_x, y=center_y, z=1.5, speed=0.5, frame_id='aruco_map')
    rospy.sleep(5)
    land()