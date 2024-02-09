#!/usr/bin/env python3
import math
import copy
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from collections import deque

import rclpy
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped, Pose
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped


class Vertex(object):
    def __init__(self, pos=None, parent=None):
        self.pos = pos
        self.parent = parent


class RRT_Stanley(Node):
    def __init__(self):
        super().__init__("rrt_stanley_node")

        self.declare_parameter("waypoints_path", "/home/f1tenth/f1tenth_ws_waterloo/src/stanley_avoidance/racelines/traj_race_cl-2023-12-13_mincurv_iqp_right.csv")
        self.declare_parameter("waypoints_path_2nd", "/home/f1tenth/f1tenth_ws_waterloo/src/stanley_avoidance/racelines/traj_race_cl-2023-12-13_mincurv_iqp_right.csv")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/pf/pose/odom")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("rviz_current_waypoint_topic", "/current_waypoint")
        self.declare_parameter("rviz_lookahead_waypoint_topic", "/lookahead_waypoint")
        self.declare_parameter("rrt_stanley_path_topic", "/rrt_stanley_path")
        self.declare_parameter("rrt_stanley_path_array_topic", "/rrt_stanley_path_array")
        self.declare_parameter("occupancy_grid_topic", "/occupancy_grid")

        self.declare_parameter("grid_width_meters", 6.0)
        self.declare_parameter("K_p", 3.5)     # default: 10(yaml:10) 3.5  5.0: a bit high
        self.declare_parameter("K_i", 0.01)   # default: 0.05  0.01:good  0.001
        self.declare_parameter("K_d", 0.05)   # default: 0.05              0.05
        self.declare_parameter("K_p_obstacle", 0.8) # default: 0.8(yaml:0.8)
        self.declare_parameter("K_E", 1.0)    # default: 1.0 (yaml:1.0)
        self.declare_parameter("K_H", 0.0)    # default: 0.0 (yaml:0.5)
        self.declare_parameter("crosstrack_error_offset", 0.1)  # default: 0.1
        self.declare_parameter("min_lookahead", 1.5)            # default: 1.0(yaml:1.5) 1.5   1.75  1.75
        self.declare_parameter("max_lookahead", 1.75)            # default: 3.0(yaml:2.5)  1.5  1.75   2.0
        self.declare_parameter("min_lookahead_speed", 0.5)      # default: 1.0(yaml:0.5)
        self.declare_parameter("max_lookahead_speed", 1.0)      # default: 1.0(yaml:1.0)
        self.declare_parameter("interpolation_distance", 0.05)  
        self.declare_parameter("velocity_min", 0.5)             # default: 0.5(yaml:0.5)
        self.declare_parameter("velocity_max", 1.0)             # default: 1.0(yaml:1.5)
        self.declare_parameter("velocity_percentage", 0.25)      # default: 0.2(yaml: 0.8) 0.5: not so good 0.25 0.35: 2 fast
        self.declare_parameter("velocity_percentage_rrt", 0.15)   # 0.15
        self.declare_parameter("steering_limit", 25.0)          # default: 25(yaml:25)
        self.declare_parameter("cells_per_meter", 10)   
        self.declare_parameter("lane_number", 0)
         
        self.waypoints_world_path = str(self.get_parameter("waypoints_path").value)
        self.waypoints_world_path_2nd = str(self.get_parameter("waypoints_path_2nd").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.drive_topic = str(self.get_parameter("drive_topic").value)
        self.rviz_current_waypoint_topic = str(self.get_parameter("rviz_current_waypoint_topic").value)
        self.rviz_lookahead_waypoint_topic = str(self.get_parameter("rviz_lookahead_waypoint_topic").value)
        self.rrt_stanley_path_topic = str(self.get_parameter("rrt_stanley_path_topic").value)
        self.rrt_stanley_path_array_topic = str(self.get_parameter("rrt_stanley_path_array_topic").value)
        self.occupancy_grid_topic = str(self.get_parameter("occupancy_grid_topic").value)
        self.L = float(self.get_parameter("max_lookahead").value)
        self.grid_width_meters = float(self.get_parameter("grid_width_meters").value)
        self.K_E = float(self.get_parameter("K_E").value)
        self.K_H = float(self.get_parameter("K_H").value)
        self.K_p = float(self.get_parameter("K_p").value)
        self.K_i = float(self.get_parameter("K_i").value)
        self.K_d = float(self.get_parameter("K_d").value)
        self.crosstrack_error_offset = str(self.get_parameter("crosstrack_error_offset").value)
        self.K_p_obstacle = float(self.get_parameter("K_p_obstacle").value)
        self.interpolation_distance = float(self.get_parameter("interpolation_distance").value)
        self.velocity_min = float(self.get_parameter("velocity_min").value)
        self.velocity_max = float(self.get_parameter("velocity_max").value)
        self.velocity_percentage = float(self.get_parameter("velocity_percentage").value)
        self.velocity_percentage_rrt = float(self.get_parameter("velocity_percentage_rrt").value)
        self.steering_limit = float(self.get_parameter("steering_limit").value)
        self.CELLS_PER_METER = int(self.get_parameter("cells_per_meter").value)
        self.lane_number = int(self.get_parameter("lane_number").value)  # Dynamically change lanes
     
        min_lookahead = float(self.get_parameter("min_lookahead").value)
        max_lookahead = float(self.get_parameter("max_lookahead").value)
        min_lookahead_speed = float(self.get_parameter("min_lookahead_speed").value)
        max_lookahead_speed = float(self.get_parameter("max_lookahead_speed").value)

        # hyper-parameters
        self.populate_free = True
        self.waypoint_utils = WaypointUtils(
            node=self,
            L=self.L,
            interpolation_distance=self.interpolation_distance,
            filepath=self.waypoints_world_path,
            min_lookahead=min_lookahead,
            max_lookahead=max_lookahead,
            min_lookahead_speed=min_lookahead_speed,
            max_lookahead_speed=max_lookahead_speed,
            filepath_2nd=self.waypoints_world_path_2nd,
        )

        self.get_logger().info(f"Loaded {len(self.waypoint_utils.waypoints_world)} waypoints")
        self.utils = Utils()

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 1)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 1)

        # publishers
        self.create_timer(1.0, self.timer_callback)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.current_waypoint_pub = self.create_publisher(Marker, self.rviz_current_waypoint_topic, 10)
        self.waypoint_pub = self.create_publisher(Marker, self.rviz_lookahead_waypoint_topic, 10)
        self.rrt_stanley_path_pub = self.create_publisher(Marker, self.rrt_stanley_path_topic, 10)
        self.rrt_stanley_path_array_pub = self.create_publisher(MarkerArray, self.rrt_stanley_path_array_topic, 10)
        self.occupancy_grid_pub = self.create_publisher(OccupancyGrid, self.occupancy_grid_topic, 10)

        # constants
        self.MAX_RANGE = self.L - 0.1
        self.MIN_ANGLE = np.radians(0)
        self.MAX_ANGLE = np.radians(180)
        # fov = 270. One side = fov/2 = 135. 135-45 = 90 (for occupancy grid)
        self.ANGLE_OFFSET = np.radians(45)
        self.IS_OCCUPIED = 100
        self.IS_FREE = 0
        self.MAX_RRT_ITER = 100
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.previous_error3 = 0.0

        # class variables
        self.grid_height = int(self.L * self.CELLS_PER_METER)
        self.grid_width = int(self.grid_width_meters * self.CELLS_PER_METER)
        self.CELL_Y_OFFSET = (self.grid_width // 2) - 1
        self.occupancy_grid = np.full(shape=(self.grid_height, self.grid_width), fill_value=-1, dtype=int)
        self.current_pose = None
        # from the laser frame, approx front wheelbase. Needed for stanley controller
        self.current_pose_wheelbase_front = None
        self.goal_pos = None
        self.closest_wheelbase_rear_point = None
        self.obstacle_detected = False
        self.target_velocity = 0.0

    def timer_callback(self):
        self.waypoints_world_path = str(self.get_parameter("waypoints_path").value)
        self.K_E = float(self.get_parameter("K_E").value)
        self.K_H = float(self.get_parameter("K_H").value)
        self.K_p = float(self.get_parameter("K_p").value)
        self.K_i = float(self.get_parameter("K_i").value)
        self.K_d = float(self.get_parameter("K_d").value)
        self.crosstrack_error_offset = float(self.get_parameter("crosstrack_error_offset").value)
        self.K_p_obstacle = float(self.get_parameter("K_p_obstacle").value)
        self.interpolation_distance = int(self.get_parameter("interpolation_distance").value)
        self.velocity_min = float(self.get_parameter("velocity_min").value)
        self.velocity_max = float(self.get_parameter("velocity_max").value)
        self.velocity_percentage = float(self.get_parameter("velocity_percentage").value)
        self.velocity_percentage_rrt = float(self.get_parameter("velocity_percentage_rrt").value)
        self.steering_limit = float(self.get_parameter("steering_limit").value)
        self.CELLS_PER_METER = int(self.get_parameter("cells_per_meter").value)

        self.waypoint_utils.min_lookahead = float(self.get_parameter("min_lookahead").value)
        self.waypoint_utils.max_lookahead = float(self.get_parameter("max_lookahead").value)
        self.waypoint_utils.min_lookahead_speed = float(self.get_parameter("min_lookahead_speed").value)
        self.waypoint_utils.max_lookahead_speed = float(self.get_parameter("max_lookahead_speed").value)

        self.waypoint_utils.lane_number = int(self.get_parameter("lane_number").value)  # Dynamically change lanes

    def local_to_grid(self, x, y):
        i = int(x * -self.CELLS_PER_METER + (self.grid_height - 1))
        j = int(y * -self.CELLS_PER_METER + self.CELL_Y_OFFSET)
        return (i, j)

    def local_to_grid_parallel(self, x, y):
        i = np.round(x * -self.CELLS_PER_METER + (self.grid_height - 1)).astype(int)
        j = np.round(y * -self.CELLS_PER_METER + self.CELL_Y_OFFSET).astype(int)
        return i, j

    def grid_to_local(self, point):
        i, j = point[0], point[1]
        x = (i - (self.grid_height - 1)) / -self.CELLS_PER_METER
        y = (j - self.CELL_Y_OFFSET) / -self.CELLS_PER_METER
        return (x, y)

    def odom_callback(self, pose_msg: Odometry):
        """
        The pose callback when subscribed to particle filter's inferred pose
        """
        # determine pose data type (sim vs. car)
        self.current_pose = pose_msg.pose.pose

        # TOO SLOW
        # to_frame_rel = "ego_racecar/base_link"
        # from_frame_rel = "ego_racecar/laser_model"

        # if self.base_link_to_laser_tf is None: # static tf, so only need to lookup once
        #     try:
        #         self.base_link_to_laser_tf = self.tf_buffer.lookup_transform(
        #                 to_frame_rel,
        #                 from_frame_rel,
        #                 rclpy.time.Time())
        #     except TransformException as ex:
        #         self.get_logger().info(
        #             f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
        #         return

        # self.get_logger().info(str(self.base_link_to_laser_tf))

        current_pose_quaternion = np.array(
            [
                self.current_pose.orientation.x,
                self.current_pose.orientation.y,
                self.current_pose.orientation.z,
                self.current_pose.orientation.w,
            ]
        )

        # 33cm in front of base_link
        self.current_pose_wheelbase_front = Pose()
        current_pose_xyz = R.from_quat(current_pose_quaternion).apply((0.33, 0, 0)) + (
            self.current_pose.position.x,
            self.current_pose.position.y,
            0,
        )
        self.current_pose_wheelbase_front.position.x = current_pose_xyz[0]
        self.current_pose_wheelbase_front.position.y = current_pose_xyz[1]
        self.current_pose_wheelbase_front.position.z = current_pose_xyz[2]
        self.current_pose_wheelbase_front.orientation = self.current_pose.orientation

        # obtain pure pursuit waypoint (base_link frame)
        self.closest_wheelbase_rear_point, self.target_velocity = self.waypoint_utils.get_closest_waypoint_with_velocity(
            self.current_pose
        )

        self.utils.draw_marker(
            pose_msg.header.frame_id,
            pose_msg.header.stamp,
            self.closest_wheelbase_rear_point,
            self.current_waypoint_pub,
            color="blue",
        )

        # self.current_velocity = np.sqrt(pose_msg.twist.twist.linear.x**2 +  pose_msg.twist.twist.linear.y**2)

        self.goal_pos, goal_pos_world = self.waypoint_utils.get_waypoint(self.current_pose, self.target_velocity)
        self.get_logger().info(f"Target velocity: {self.target_velocity:.2f}")
        self.utils.draw_marker(
            pose_msg.header.frame_id, 
            pose_msg.header.stamp, 
            goal_pos_world, 
            self.waypoint_pub, 
            color="red")

    def populate_occupancy_grid(self, ranges, angle_increment):
        """
        Populate occupancy grid using lidar scans and save
        the data in class member variable self.occupancy_grid.

        Optimization performed to improve the speed at which we generate the occupancy grid.

        Args:
            scan_msg (LaserScan): message from lidar scan topic
        """
        # reset empty occupacny grid (-1 = unknown)

        self.occupancy_grid = np.full(shape=(self.grid_height, self.grid_width), fill_value=self.IS_FREE, dtype=int)

        ranges = np.array(ranges)
        indices = np.arange(len(ranges))
        thetas = (indices * angle_increment) - self.ANGLE_OFFSET
        xs = ranges * np.sin(thetas)
        ys = ranges * np.cos(thetas) * -1

        i, j = self.local_to_grid_parallel(xs, ys)

        occupied_indices = np.where((i >= 0) & (i < self.grid_height) & (j >= 0) & (j < self.grid_width))
        self.occupancy_grid[i[occupied_indices], j[occupied_indices]] = self.IS_OCCUPIED

    # NYI
    def publish_occupancy_grid(self, frame_id, stamp):
        """
        Publish populated occupancy grid to ros2 topic
        Args:
            scan_msg (LaserScan): message from lidar scan topic
        """
        oc = OccupancyGrid()
        oc.header.frame_id = frame_id
        oc.header.stamp = stamp
        oc.info.origin.position.y -= ((self.grid_width / 2) + 1) / self.CELLS_PER_METER
        oc.info.width = self.grid_height
        oc.info.height = self.grid_width
        oc.info.resolution = 1 / self.CELLS_PER_METER
        oc.data = np.fliplr(np.rot90(self.occupancy_grid, k=1)).flatten().tolist()
        self.occupancy_grid_pub.publish(oc)

    def convolve_occupancy_grid(self):
        kernel = np.ones(shape=[2, 2])
        self.occupancy_grid = signal.convolve2d(
            self.occupancy_grid.astype("int"), 
            kernel.astype("int"), 
            boundary="symm", 
            mode="same"
        )
        self.occupancy_grid = np.clip(self.occupancy_grid, -1, 100)

    def drive_to_target(self, point, K_p):
        """
        Using the pure pursuit derivation

        Improvement is that we make the point closer when the car is going at higher speeds

        """
        # calculate curvature/steering angle
        L = np.linalg.norm(point)
        y = point[1]
        angle = K_p * (2 * y) / (L**2)
        angle = np.clip(angle, -np.radians(self.steering_limit), np.radians(self.steering_limit))

        # determine velocity
        if self.obstacle_detected and self.velocity_percentage_rrt > 0.0:
            if np.degrees(angle) < 10.0:
                velocity = self.velocity_max
            elif np.degrees(angle) < 20.0:
                velocity = (self.velocity_max + self.velocity_min) / 2
            else:
                velocity = self.velocity_min

        else:
            # Set velocity to velocity of racing line
            velocity = self.target_velocity * self.velocity_percentage_rrt

        #logit = np.clip(
        #    (2.0 / (1.0 + np.exp(-5 * (np.abs(np.degrees(angle)) / self.steering_limit)))) - 1.0,
        #    1.0
        #)
        #velocity = self.velocity_max - (logit * (self.velocity_max - self.velocity_min))

        # publish drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = angle
        self.get_logger().info(
            f"Obstacle: {self.obstacle_detected} ... lookahead: {self.waypoint_utils.L:.2f} ... index: {self.waypoint_utils.index} ... Speed: {velocity:.2f}m/s ... Steering Angle: {np.degrees(angle):.2f} ... K_p: {self.K_p} ... K_p_obstacle: {self.K_p_obstacle} ... velocity_percentage_rrt: {self.velocity_percentage_rrt:.2f}"
        )
        self.drive_pub.publish(drive_msg)

    def drive_to_target_stanley(self):
        """
        Using the stanley method derivation: https://stevengong.co/notes/Stanley-Method

        Might get the best out of both worlds for responsiveness, and less oscillations compared to pure pursuit.
        """
        K_V = 0
        # calculate curvature/steering angle
        closest_wheelbase_front_point_car, closest_wheelbase_front_point_world = self.waypoint_utils.get_waypoint_stanley(
            self.current_pose_wheelbase_front
        )

        path_heading = math.atan2(
            closest_wheelbase_front_point_world[1] - self.closest_wheelbase_rear_point[1],
            closest_wheelbase_front_point_world[0] - self.closest_wheelbase_rear_point[0],
        )
        current_heading = math.atan2(
            self.current_pose_wheelbase_front.position.y - self.current_pose.position.y,
            self.current_pose_wheelbase_front.position.x - self.current_pose.position.x,
        )

        if current_heading < 0:
            current_heading += 2 * math.pi
        if path_heading < 0:
            path_heading += 2 * math.pi

        # calculate the errors
        crosstrack_error = float(closest_wheelbase_front_point_car[1]) + float(self.crosstrack_error_offset)
        #crosstrack_error = math.atan2(
        #    self.K_E * closest_wheelbase_front_point_car[1], K_V + self.target_velocity
        #)  # y value in car frame
        heading_error = path_heading - current_heading
        if heading_error > math.pi:
            heading_error -= 2 * math.pi
        elif heading_error < -math.pi:
            heading_error += 2 * math.pi

        heading_error *= self.K_H
        
        error1 = (self.K_p * crosstrack_error)
        error2 = (self.K_d * (crosstrack_error - self.previous_error)/0.01)
        error3 = (self.previous_error3 + (self.K_i*crosstrack_error*0.01))
        error = error1 + error2 + error3
        cte_front = math.atan2(error,self.target_velocity)
        angle = cte_front + heading_error
        
        # Update integral of crosstrack error
        #self.integral_error += crosstrack_error
        # Calculate derivative of crosstrack error
        #derivative_error = crosstrack_error - self.previous_error
        # Compute the steering angle using PID control
        #self.get_logger().info(f"I'm reaching the PID")
        #angle = self.K_p * crosstrack_error + self.K_i * self.integral_error + self.K_d * derivative_error #+ heading_error
        # Update previous crosstrack error
        self.previous_error = crosstrack_error
        self.previous_error3 = error3
        self.get_logger().info(
            f"K_P: {self.K_p} ... K_D: {self.K_d} ... K_I: {self.K_i} ... K_H: {self.K_H} ... K_E: {self.K_E} ... crosstrack_error_offset: {self.crosstrack_error_offset}"
        )
      
        self.get_logger().info(
            f"heading_error: {heading_error:.2f} ... crosstrack_error: {crosstrack_error:.2f} angle: {np.degrees(angle):.2f}"
        )
        self.get_logger().info(f"current_heading: {current_heading:.2f} ... path_heading: {path_heading:.2f}")

        angle = np.clip(angle, -np.radians(self.steering_limit), np.radians(self.steering_limit))
        self.get_logger().info(f"angle: {np.degrees(angle):.2f}")

        velocity = self.target_velocity * self.velocity_percentage
        self.get_logger().info(f"velocity: {velocity:.2f}")
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = angle
        self.drive_pub.publish(drive_msg)

    def scan_callback(self, scan_msg):
        """
        LaserScan callback, update occupancy grid and perform RRT

        Args:
            scan_msg (LaserScan): incoming message from subscribed topic
        """
        # make sure we obtain initial pose and goal point
        if (self.current_pose is None) or (self.goal_pos is None):
            return

        # populate occupancy grid
        self.populate_occupancy_grid(scan_msg.ranges, scan_msg.angle_increment)
        self.convolve_occupancy_grid()
        self.publish_occupancy_grid(scan_msg.header.frame_id, scan_msg.header.stamp)

       # convert position to occupancy grid indices
        #path_local = []

        current_pos = np.array(self.local_to_grid(0, 0))
        goal_pos = np.array(self.local_to_grid(self.goal_pos[0], self.goal_pos[1]))
        MARGIN = int(self.CELLS_PER_METER * 0.25) # 0.15m margin on each side, since the car is ~0.3m wide
        
        # resample a close point if our goal point is occupied
        if self.check_collision_new(current_pos, goal_pos, margin=MARGIN):
        # if self.check_collision(current_pos, goal_pos):
        #if self.occupancy_grid[goal_pos] == self.IS_OCCUPIED:
            self.obstacle_detected = True

            # activate rrt and get path planed in occupancy grid space
            path_grid = self.rrt()

            # convert path from grid to local coordinates
            path_local = [self.grid_to_local(point) for point in path_grid]
            
            self.get_logger().info(f"Path: {path_local}")
            
            if len(path_local) < 2:
                return
            
            # navigate to first node in tree
            self.drive_to_target(path_local[1], self.K_p_obstacle)
        else:
            self.obstacle_detected = False
            path_local = []
            path_local = [self.grid_to_local(current_pos)]
            target = self.grid_to_local(goal_pos)
            path_local.append(target)
            self.drive_to_target_stanley()             

        # Visualization
        self.utils.draw_marker_array(
            scan_msg.header.frame_id, 
            scan_msg.header.stamp, 
            path_local, 
            self.rrt_stanley_path_array_pub)
        self.utils.draw_lines(
            scan_msg.header.frame_id, 
            scan_msg.header.stamp,
            path_local, 
            self.rrt_stanley_path_pub
            )



    def rrt(self):
        # convert position to occupancy grid indices
        current_pos = self.local_to_grid(0, 0)
        goal_pos = self.local_to_grid(self.goal_pos[0], self.goal_pos[1])

        # resample a close point if our goal point is occupied
        if self.occupancy_grid[goal_pos] == self.IS_OCCUPIED:
            i, j = self.sample()
            while np.linalg.norm(np.array([i, j]) - np.array(goal_pos)) > 5:
                i, j = self.sample()
            goal_pos = (i, j)

        # initialize start and goal trees
        T_start = [Vertex(current_pos)]
        T_goal = [Vertex(goal_pos)]

        # start rrt algorithm
        for itn in range(self.MAX_RRT_ITER):
            # sample from free space
            pos_sampled = self.sample()

            # attempt to expand tree using sampled point
            T_start, success_start = self.expand_tree(T_start, pos_sampled, check_closer=True)
            T_goal, success_goal = self.expand_tree(T_goal, pos_sampled)

            # if sampled point can reach both T_start and T_goal
            # get path from start to goal and return
            if success_start and success_goal:
                path = self.find_path(T_start, T_goal, pruning=True)    # default one: pruning is True
                return path
        return []
        
    def sample(self):
        """
        Randomly sample the free space in occupancy grid, and returns its index.
        If free space has already been populated then just check if sampled
        cell is free, else do fast voxel traversal for each sampling.

        Returns:
            (i, j) (int, int): index of free cell in occupancy grid
        """
        if self.populate_free:
            i, j = np.random.randint(self.grid_height), np.random.randint(self.grid_width)
            while self.occupancy_grid[i, j] != self.IS_FREE:
                i, j = np.random.randint(self.grid_height), np.random.randint(self.grid_width)
        else:
            free = False
            while not free:
                i, j = np.random.randint(self.grid_height), np.random.randint(self.grid_width)
                free = True
                for cell in self.utils.traverse_grid(self.local_to_grid(0, 0), (i, j)):
                    if self.occupancy_grid[cell] == self.IS_OCCUPIED:
                        free = False
                        break
        return (i, j)

    def expand_tree(self, tree, sampled_point, check_closer=False):
        """
        Attempts to expand tree using the sampled point by
        checking if it causes collision and if the new node
        brings the car closer to the goal.

        Args:
            tree ([]): current RRT tree
            sampled_point: cell sampled in occupancy grid free space
            check_closer: check if sampled point brings car closer
        Returns:
            tree ([]): expanded RRT tree
            success (bool): whether tree was successfully expanded
        """
        # get closest node to sampled point in tree
        idx_nearest = self.nearest(tree, sampled_point)
        pos_nearest = tree[idx_nearest].pos

        # check if nearest node -> sampled node causes collision
        collision = self.check_collision(sampled_point, pos_nearest)

        # check if sampeld point bring car closer to goal compared to the nearest node
        is_closer = self.is_closer(sampled_point, pos_nearest) if check_closer else True

        # if p_free -> p_nearest causes no collision
        # then add p_free as child of p_nearest in T_start
        if is_closer and (not collision):
            tree.append(Vertex(sampled_point, idx_nearest))

        return tree, (is_closer and (not collision))

    def nearest(self, tree, sampled_cell):
        """
        Return the nearest node on the tree to the sampled point

        Args:
            tree ([]): the current RRT tree
            sampled_cell (i,j): cell sampled in occupancy grid free space
        Returns:
            nearest_indx (int): index of neareset node on the tree
        """
        nearest_indx = -1
        nearest_dist = np.Inf
        for idx, node in enumerate(tree):
            dist = np.linalg.norm(np.array(sampled_cell) - np.array(node.pos))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_indx = idx
        return nearest_indx

    def is_closer(self, sampled_pos, nearest_pos):
        """
        Checks if the new sampled node brings the car closer
        to the goal point then the nearest existing node on the tree

        Args:
            sampled_pos (i, j): index of sampled pos in occupancy grid
            nearest_pos (i, j): index of nearest pos in occupancy grid
        Returns:
            is_closer (bool): whether the sampled pos brings the car closer
        """
        a = self.grid_to_local(sampled_pos)
        b = self.grid_to_local(nearest_pos)
        return np.linalg.norm(a - self.goal_pos[:2]) < np.linalg.norm(b - self.goal_pos[:2])

    def check_collision(self, cell_a, cell_b):
        """
        Checks whether the path between two cells
        in the occupancy grid is collision free.

        Args:
            cell_a (i, j): index of cell a in occupancy grid
            cell_b (i, j): index of cell b in occupancy grid
        Returns:
            collision (bool): whether path between two cells would cause collision
        """
        for cell in self.utils.traverse_grid(cell_a, cell_b):
            if (cell[0] * cell[1] < 0) or (cell[0] >= self.grid_height) or (cell[1] >= self.grid_width):
                continue
            if self.occupancy_grid[cell] == self.IS_OCCUPIED:
                return True
        return False

    def check_collision_new(self, cell_a, cell_b, margin=0):
        """
        Checks whether the path between two cells
        in the occupancy grid is collision free.

        The margin is done by checking if adjacent cells are also free.

        One of the issues is that if the starting cell is next to a wall, then it already considers there to be a collision.
        See check_collision_loose


        Args:
            cell_a (i, j): index of cell a in occupancy grid
            cell_b (i, j): index of cell b in occupancy grid
            margin (int): margin of safety around the path
        Returns:
            collision (bool): whether path between two cells would cause collision
        """
        for i in range(-margin, margin + 1):  # for the margin, check
            cell_a_margin = (cell_a[0], cell_a[1] + i)
            cell_b_margin = (cell_b[0], cell_b[1] + i)
            for cell in self.utils.traverse_grid(cell_a_margin, cell_b_margin):
                if (cell[0] * cell[1] < 0) or (cell[0] >= self.grid_height) or (cell[1] >= self.grid_width):
                    continue
                try:
                    if self.occupancy_grid[cell] == self.IS_OCCUPIED:
                        return True
                except:
                    self.get_logger().info(f"Sampled point is out of bounds: {cell}")
                    return True
        return False
        
    def find_path(self, T_start, T_goal, pruning=True):
        """
        Returns a path as a list of Nodes connecting the starting point to
        the goal once the latest added node is close enough to the goal

        Args:
            tree ([]): current tree as a list of Nodes
            latest_added_node (Node): latest added node in the tree
        Returns:
            path ([]): valid path as a list of Nodes
        """
        # traverse up T_start to obtain path to sampled point
        node = T_start[-1]
        path_start = [node.pos]
        while node.parent is not None:
            node = T_start[node.parent]
            path_start.append(node.pos)

        # traverse up T_goal to obtain path to sampled point
        node = T_goal[-1]
        path_goal = [node.pos]
        while node.parent is not None:
            node = T_goal[node.parent]
            path_goal.append(node.pos)

        # return path
        path = np.array(path_start[::-1] + path_goal[1:])

        # pruning if enabled
        if pruning:
            sub_paths = []
            for i in range(len(path) - 2):
                sub_path = path
                for j in range(i + 2, len(path)):
                    if not self.check_collision(path[i], path[j]):
                        sub_path = np.vstack((path[: i + 1], path[j:]))
                sub_paths.append(sub_path)

            costs = np.array([np.linalg.norm(p[1:] - p[:-1]).sum() for p in sub_paths])
            path = sub_paths[np.argmin(costs)]
        return path


class Utils:
    def __init__(self):
        pass

    def draw_marker(self, frame_id, stamp, position, publisher, color="red", id=0):
        if position is None:
            return
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.id = id
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 1.0
        if color == "red":
            marker.color.r = 1.0
        elif color == "green":
            marker.color.g = 1.0
        elif color == "blue":
            marker.color.b = 1.0
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = 0.0
        publisher.publish(marker)

    def draw_marker_array(self, frame_id, stamp, positions, publisher):
        marker_array = MarkerArray()
        for i, position in enumerate(positions):
            if position is None:
                continue
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.id = i
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.position.x = position[0]
            marker.pose.position.y = position[1]
            marker.pose.position.z = 0.0
            marker.lifetime = Duration(seconds=0.1).to_msg()
            marker_array.markers.append(marker)
        publisher.publish(marker_array)

    def draw_lines(self, frame_id, stamp, path, publisher):
        points = []
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            point = Point()
            point.x = a[0]
            point.y = a[1]
            points.append(copy.deepcopy(point))
            point.x = b[0]
            point.y = b[1]
            points.append(copy.deepcopy(point))

        line_list = Marker()
        line_list.header.frame_id = frame_id
        line_list.header.stamp = stamp
        line_list.id = 0
        line_list.type = line_list.LINE_LIST
        line_list.action = line_list.ADD
        line_list.scale.x = 0.1
        line_list.color.a = 1.0
        line_list.color.r = 0.0
        line_list.color.g = 1.0
        line_list.color.b = 0.0
        line_list.points = points
        publisher.publish(line_list)

    def traverse_grid(self, start, end):
        """
        Bresenham's line algorithm for fast voxel traversal

        CREDIT TO: Rogue Basin
        CODE TAKEN FROM: http://www.roguebasin.com/index.php/Bresenham%27s_Line_Algorithm
        """
        # Setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1

        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)

        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        # Swap start and end points if necessary and store swap state
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1

        # Calculate error
        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1

        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
        return points


class WaypointUtils:
    def __init__(
        self,
        node,
        L=1.7,
        interpolation_distance=None,
        filepath="/f1tenth_ws/racelines/e7_floor5.csv",
        min_lookahead=0.5,
        max_lookahead=3.0,
        min_lookahead_speed=3.0,
        max_lookahead_speed=6.0,
        filepath_2nd="/f1tenth_ws/racelines/e7_floor5.csv",
        lane_number=0,
    ):

        self.node = node
        self.L = L  # dynamic lookahead distance
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.min_lookahead_speed = min_lookahead_speed
        self.max_lookahead_speed = max_lookahead_speed

        self.waypoints_world, self.velocities = self.load_and_interpolate_waypoints(
            file_path=filepath, interpolation_distance=interpolation_distance
        )

        # For competition, where I want to customize the lanes that I am using
        self.lane_number = lane_number
        self.waypoints_world_2nd, self.velocities_2nd = self.load_and_interpolate_waypoints(
            file_path=filepath_2nd, interpolation_distance=interpolation_distance
        )

        self.index = 0
        self.velocity_index = 0
        print(f"Loaded {len(self.waypoints_world)} waypoints")

    def transform_waypoints(self, waypoints, car_position, pose):
        # translation
        waypoints = waypoints - car_position

        # rotation
        quaternion = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        waypoints = R.inv(R.from_quat(quaternion)).apply(waypoints)

        return waypoints

    def load_and_interpolate_waypoints(self, file_path, interpolation_distance=0.05):
        # Read waypoints from csv, first two columns are x and y, third column is velocity
        # Exclude last row, because that closes the loop
        points = np.genfromtxt(file_path, delimiter=",")[:, :2]
        velocities = np.genfromtxt(file_path, delimiter=",")[:, 2]

        # Add first point as last point to complete loop
        self.node.get_logger().info(str(velocities))

        # interpolate, not generally needed because interpolation can be done with the solver, where you feed in target distance between points
        if interpolation_distance != 0 and interpolation_distance is not None:
            # Calculate the cumulative distances between points
            distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
            cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

            # Calculate the number of segments based on the desired distance threshold
            total_distance = cumulative_distances[-1]
            segments = int(total_distance / interpolation_distance)

            # Linear length along the line
            distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
            # Normalize distance between 0 and 1
            distance = np.insert(distance, 0, 0) / distance[-1]

            # Interpolate
            alpha = np.linspace(0, 1, segments)
            interpolator = interp1d(distance, points, kind="slinear", axis=0)
            interpolated_points = interpolator(alpha)

            # Interpolate velocities
            velocity_interpolator = interp1d(distance, velocities, kind="slinear")
            interpolated_velocities = velocity_interpolator(alpha)

            # Add z-coordinate to be 0
            interpolated_points = np.hstack((interpolated_points, np.zeros((interpolated_points.shape[0], 1))))
            assert len(interpolated_points) == len(interpolated_velocities)
            return interpolated_points, interpolated_velocities

        else:
            # Add z-coordinate to be 0
            points = np.hstack((points, np.zeros((points.shape[0], 1))))
            return points, velocities

    def get_closest_waypoint_with_velocity(self, pose):
        # get current position of car
        if pose is None:
            return

        position = (pose.position.x, pose.position.y, 0)

        # transform way-points from world to vehicle frame of reference
        if self.lane_number == 0:
            waypoints_car = self.transform_waypoints(self.waypoints_world, position, pose)
        else:
            waypoints_car = self.transform_waypoints(self.waypoints_world_2nd, position, pose)

        # get distance from car to all waypoints
        distances = np.linalg.norm(waypoints_car, axis=1)

        # get indices of waypoints sorted by ascending distance
        self.velocity_index = np.argmin(distances)

        if self.lane_number == 0:
            return self.waypoints_world[self.velocity_index], self.velocities[self.velocity_index]
        else:
            return self.waypoints_world_2nd[self.velocity_index], self.velocities_2nd[self.velocity_index]

    def get_waypoint_stanley(self, pose):
        # get current position of car
        if pose is None:
            return
        position = (pose.position.x, pose.position.y, 0)

        # transform way-points from world to vehicle frame of reference
        if self.lane_number == 0:
            waypoints_car = self.transform_waypoints(self.waypoints_world, position, pose)
        else:
            waypoints_car = self.transform_waypoints(self.waypoints_world_2nd, position, pose)

        # get distance from car to all waypoints
        distances = np.linalg.norm(waypoints_car, axis=1)

        # get indices of waypoints sorted by ascending distance
        index = np.argmin(distances)

        if self.lane_number == 0:
            return waypoints_car[index], self.waypoints_world[index]
        else:
            return waypoints_car[index], self.waypoints_world_2nd[index]

    def get_waypoint(self, pose, target_velocity, fixed_lookahead=None):
        # get current position of car
        if pose is None:
            return
        position = (pose.position.x, pose.position.y, 0)

        # transform way-points from world to vehicle frame of reference
        if self.lane_number == 0:
            waypoints_car = self.transform_waypoints(self.waypoints_world, position, pose)
        else:
            waypoints_car = self.transform_waypoints(self.waypoints_world_2nd, position, pose)

        # get distance from car to all waypoints
        distances = np.linalg.norm(waypoints_car, axis=1)

        # get indices of waypoints that are within L, sorted by descending distance
        # Use dynamic lookahead for this part

        if fixed_lookahead:
            self.L = fixed_lookahead
        else:
            # Lookahead is proportional to velocity
            self.L = min(
                max(
                    self.min_lookahead,
                    self.min_lookahead
                    + (self.max_lookahead - self.min_lookahead)
                    * (target_velocity - self.min_lookahead_speed)
                    / (self.max_lookahead_speed - self.min_lookahead_speed),
                ),
                self.max_lookahead,
            )

        indices_L = np.argsort(np.where(distances < self.L, distances, -1))[::-1]

        # set goal point to be the farthest valid waypoint within distance L
        for i in indices_L:
            # check waypoint is in front of car
            x = waypoints_car[i][0]
            if x > 0:
                self.index = i
                if self.lane_number == 0:
                    return waypoints_car[self.index], self.waypoints_world[self.index]
                else:
                    return waypoints_car[self.index], self.waypoints_world_2nd[self.index]
        return None, None


def main(args=None):
    rclpy.init(args=args)
    print("RRT and Stanley Avoidance Initialized")
    rrt_stanley_node = RRT_Stanley()
    rclpy.spin(rrt_stanley_node)

    rrt_stanley_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
