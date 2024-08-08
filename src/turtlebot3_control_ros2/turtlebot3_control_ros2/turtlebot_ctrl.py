import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
from math import atan2, sqrt, pi
from collections import deque

class TurtlebotCtrl(Node):
    def __init__(self):
        super().__init__("TurtlebotCtrl")

        self.laser = LaserScan()
        self.odom = Odometry()

        self.map = np.array([	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        ])

        self.publish_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.subscriber_odom = self.create_subscription(Odometry, "/odom", self.callback_odom, 10)
        self.subscriber_laser = self.create_subscription(LaserScan, "/scan", self.callback_laser, 10)
        self.timer = self.create_timer(0.5, self.cmd_vel_pub)
        self.target = None

    def bfs(self, start):
        queue = deque([start])
        visited = set()
        visited.add(start)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y = queue.popleft()
            if self.map[x][y] == 1:
                return (x, y)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map.shape[0] and 0 <= ny < self.map.shape[1] and (nx, ny) not in visited and self.map[nx][ny] == 1:
                    queue.append((nx, ny))
                    visited.add((nx, ny))
        return None

    def cmd_vel_pub(self):
        map_resolution = 4

        index_x = -int(self.odom.pose.pose.position.x * map_resolution)
        index_y = -int(self.odom.pose.pose.position.y * map_resolution)

        index_x += int(self.map.shape[0] / 2)
        index_y += int(self.map.shape[0] / 2)

        if index_x < 1: index_x = 1
        if index_x > self.map.shape[0] - 1: index_x = self.map.shape[0] - 1
        if index_y < 1: index_y = 1
        if index_y > self.map.shape[0] - 1: index_y = self.map.shape[0] - 1

        if self.map[index_x][index_y] == 1:
            self.map[index_x][index_y] = 2

            self.get_logger().info("Another part reached ... percentage total reached...." + str(100 * float(np.count_nonzero(self.map == 2)) / (np.count_nonzero(self.map == 1) + np.count_nonzero(self.map == 2))))
            self.get_logger().info("Discrete Map")
            self.get_logger().info("\n" + str(self.map))

        if self.target is None or self.map[self.target[0]][self.target[1]] != 1:
            self.target = self.bfs((index_x, index_y))

        if self.target:
            target_x, target_y = self.target
            angle_to_target = atan2(target_y - index_y, target_x - index_x)
            distance_to_target = sqrt((target_x - index_x) ** 2 + (target_y - index_y) ** 2)
            self.get_logger().info("Target: " + str(self.target))
            self.get_logger().info("Distance to target: " + str(distance_to_target))
            
            twist = Twist()
            if distance_to_target > 0.1:
                twist.linear.x = 0.2
                twist.angular.z = 0.5 * (angle_to_target - self.get_yaw())
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.target = None

            front_ranges = self.laser.ranges[:30] + self.laser.ranges[-30:]
            if any(r < 0.3 for r in front_ranges):
                twist.linear.x = 0.0
                twist.angular.z = 0.5 
            else:
                twist.angular.z = 0.0 

            self.publish_cmd_vel.publish(twist)

    def get_yaw(self):
        orientation_q = self.odom.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        return atan2(siny_cosp, cosy_cosp)

    def callback_odom(self, msg):
        self.odom = msg
        
    def callback_laser(self, msg):
        self.laser = msg

def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotCtrl()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()