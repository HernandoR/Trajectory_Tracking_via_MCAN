from math import sin, cos, atan2, radians

import numpy as np
from numpy.typing import ArrayLike

from scipy.interpolate import CubicSpline

def normalize_angle(angle): 
    # Normalizes an angle to be within the range: -pi to pi
    return atan2(sin(angle), cos(angle))


class Vehicle:
    def __init__(
        self,
        path_x,
        path_y,
        throttle,
        dt,
        control_gain,
        softening_gain,
        yaw_rate_gain,
        steering_damp_gain,
        max_steer,
        c_r: float,
        c_a: float,
        wheelbase=2.96,
        overall_length=4.97,
        overall_width=1.964,
        rear_overhang=0.0,
        tyre_diameter=0.4826,
        tyre_width=0.265,
        axle_track=1.7,
    ):
        self.k = control_gain
        self.k_soft = softening_gain
        self.k_yaw_rate = yaw_rate_gain
        self.k_damp_steer = steering_damp_gain
        self.max_steer = max_steer
        self.wheelbase = wheelbase

        self.px = path_x
        self.py = path_y
        self.pyaw = self.calculate_spline_yaw(self.px, self.py)
        self.start_heading = self.pyaw[0]

        self.x = path_x[0]
        self.y = path_y[0]
        self.yaw = self.pyaw[0]
        self.crosstrack_error = None
        self.target_id = 0

        self.v = 0.0
        self.delta = 0.0
        self.omega = 0.0
        self.throttle = throttle

        self.dt = dt
        self.c_r = c_r
        self.c_a = c_a
        
        self.logs = {
            "velocity": [],
            "angVel":[],
            "trueCarPos":[]
        }

        self.rear_overhang = 0.5 * (overall_length - self.wheelbase)
        rear_axle_to_front_bumper = overall_length - rear_overhang
        centreline_to_wheel_centre = 0.5 * axle_track
        centreline_to_side = 0.5 * overall_width
        vehicle_vertices = np.array(
            [
                (-rear_overhang, centreline_to_side),
                (rear_axle_to_front_bumper, centreline_to_side),
                (rear_axle_to_front_bumper, -centreline_to_side),
                (-rear_overhang, -centreline_to_side),
            ]
        )
        half_tyre_width = 0.5 * tyre_width
        centreline_to_inwards_rim = centreline_to_wheel_centre - half_tyre_width
        centreline_to_outwards_rim = centreline_to_wheel_centre + half_tyre_width
        # Rear right wheel vertices
        wheel_vertices = np.array(
            [
                (-tyre_diameter, -centreline_to_inwards_rim),
                (tyre_diameter, -centreline_to_inwards_rim),
                (tyre_diameter, -centreline_to_outwards_rim),
                (-tyre_diameter, -centreline_to_outwards_rim),
            ]
        )
        self.outlines = np.concatenate([vehicle_vertices, [vehicle_vertices[0]]])
        self.rear_right_wheel = np.concatenate([wheel_vertices, [wheel_vertices[0]]])
        # Reflect the wheel vertices about the x-axis
        self.rear_left_wheel = self.rear_right_wheel.copy()
        self.rear_left_wheel[:, 1] *= -1
        # Translate the wheel vertices to the front axle
        front_left_wheel = self.rear_left_wheel.copy()
        front_right_wheel = self.rear_right_wheel.copy()
        front_left_wheel[:, 0] += wheelbase
        front_right_wheel[:, 0] += wheelbase
        get_face_centre = lambda vertices: np.array(
            [
                0.5 * (vertices[0][0] + vertices[2][0]),
                0.5 * (vertices[0][1] + vertices[2][1]),
            ]
        )
        # Translate front wheels to origin
        self.fr_wheel_centre = get_face_centre(front_right_wheel)
        self.fl_wheel_centre = get_face_centre(front_left_wheel)
        self.fr_wheel_origin = front_right_wheel - self.fr_wheel_centre
        self.fl_wheel_origin = front_left_wheel - self.fl_wheel_centre


    def get_logs(self):
        return self.logs
    def initialise_cubic_spline(
        self, x: ArrayLike, y: ArrayLike, ds: float, bc_type: str
    ):
        distance = np.concatenate(
            (np.zeros(1), np.cumsum(np.hypot(np.ediff1d(x), np.ediff1d(y))))
        )
        points = np.array([x, y]).T
        s = np.arange(0, distance[-1], ds)

        try:
            cs = CubicSpline(
                distance, points, bc_type=bc_type, axis=0, extrapolate=False
            )

        except ValueError as e:
            raise ValueError(
                f"{e} If you are getting a sequence error, do check if your input dataset contains consecutive duplicate(s)."
            )

        return cs, s

    def calculate_spline_yaw(
        self, x: ArrayLike, y: ArrayLike, ds: float = 0.05, bc_type: str = "natural"
    ):
        cs, s = self.initialise_cubic_spline(x, y, ds, bc_type)
        dx, dy = cs.derivative(1)(s).T
        return np.arctan2(dy, dx)

    def find_target_path_id(self, x, y, yaw):
        # Calculate position of the front axle
        fx = x + self.wheelbase * cos(yaw)
        fy = y + self.wheelbase * sin(yaw)

        dx = fx - self.px  # Find the x-axis of the front axle relative to the path
        dy = fy - self.py  # Find the y-axis of the front axle relative to the path

        d = np.hypot(dx, dy)  # Find the distance from the front axle to the path
        target_index = np.argmin(d)  # Find the shortest distance in the array

        return target_index, dx[target_index], dy[target_index], d[target_index]

    def calculate_yaw_term(self, target_index, yaw):
        yaw_error = normalize_angle(self.pyaw[target_index] - yaw)
        # yaw_error = self.pyaw[target_index] - yaw

        return yaw_error

    def calculate_crosstrack_term(self, target_velocity, yaw, dx, dy, absolute_error):
        front_axle_vector = np.array([sin(yaw), -cos(yaw)])
        nearest_path_vector = np.array([dx, dy])
        crosstrack_error = (
            np.sign(nearest_path_vector @ front_axle_vector) * absolute_error
        )

        crosstrack_steering_error = atan2(
            (self.k * crosstrack_error), (self.k_soft + target_velocity)
        )

        return crosstrack_steering_error, crosstrack_error

    def calculate_yaw_rate_term(self, target_velocity, steering_angle):
        yaw_rate_error = (
            self.k_yaw_rate * (-target_velocity * sin(steering_angle)) / self.wheelbase
        )

        return yaw_rate_error

    def calculate_steering_delay_term(
        self, computed_steering_angle, previous_steering_angle
    ):
        steering_delay_error = self.k_damp_steer * (
            computed_steering_angle - previous_steering_angle
        )

        return steering_delay_error

    def stanley_control(self, x, y, yaw, target_velocity, steering_angle):
        target_index, dx, dy, absolute_error = self.find_target_path_id(x, y, yaw)
        yaw_error = self.calculate_yaw_term(target_index, yaw)
        crosstrack_steering_error, crosstrack_error = self.calculate_crosstrack_term(
            target_velocity, yaw, dx, dy, absolute_error
        )
        yaw_rate_damping = self.calculate_yaw_rate_term(target_velocity, steering_angle)

        desired_steering_angle = (
            yaw_error + crosstrack_steering_error + yaw_rate_damping
        )

        # Constrains steering angle to the vehicle limits
        desired_steering_angle += self.calculate_steering_delay_term(
            desired_steering_angle, steering_angle
        )
        limited_steering_angle = np.clip(
            desired_steering_angle, -self.max_steer, self.max_steer
        )

        return limited_steering_angle, target_index, crosstrack_error

    def kinematic_model(
        self,
        x: float,
        y: float,
        yaw: float,
        velocity: float,
        throttle: float,
        steering_angle: float,
    ):
        # Compute the local velocity in the x-axis
        friction = velocity * (self.c_r + self.c_a * velocity)
        new_velocity = velocity + self.dt * (throttle - friction)

        # Limit steering angle to physical vehicle limits
        # steering_angle = -self.max_steer if steering_angle < -self.max_steer else self.max_steer if steering_angle > self.max_steer else steering_angle

        # Compute the angular velocity
        angular_velocity = velocity * np.tan(steering_angle) / self.wheelbase

        # Compute the final state using the discrete time model
        new_x = x + velocity * np.cos(yaw) * self.dt
        new_y = y + velocity * np.sin(yaw) * self.dt
        new_yaw = normalize_angle(yaw + angular_velocity * self.dt)

        return new_x, new_y, new_yaw, new_velocity, steering_angle, angular_velocity

    def get_rotation_matrix(_, angle: float) -> np.ndarray:
        cos_angle = cos(angle)
        sin_angle = sin(angle)

        return np.array([(cos_angle, sin_angle), (-sin_angle, cos_angle)])

    def transform(self, point: np.ndarray) -> np.ndarray:
        # Vector rotation
        point = point.dot(self.yaw_vector).T

        # Vector translation
        point[0, :] += self.x
        point[1, :] += self.y

        return point

    def plot_car(self, x: float, y: float, yaw: float, steer: float):
        self.x = x
        self.y = y

        # Rotation matrices
        self.yaw_vector = self.get_rotation_matrix(yaw)
        steer_vector = self.get_rotation_matrix(steer)

        # Rotate the wheels about its position
        front_right_wheel = self.fr_wheel_origin.copy()
        front_left_wheel = self.fl_wheel_origin.copy()
        front_right_wheel = front_right_wheel @ steer_vector
        front_left_wheel = front_left_wheel @ steer_vector
        front_right_wheel += self.fr_wheel_centre
        front_left_wheel += self.fl_wheel_centre

        outlines = self.transform(self.outlines)
        rear_right_wheel = self.transform(self.rear_right_wheel)
        rear_left_wheel = self.transform(self.rear_left_wheel)
        front_right_wheel = self.transform(front_right_wheel)
        front_left_wheel = self.transform(front_left_wheel)

        return (
            outlines,
            front_right_wheel,
            rear_right_wheel,
            front_left_wheel,
            rear_left_wheel,
        )

    def drive(self):
        # throttle = 300 #uniform(50, 200)
        self.delta, self.target_id, self.crosstrack_error = self.stanley_control(
            self.x, self.y, self.yaw, self.v, self.delta
        )
        self.x, self.y, self.yaw, self.v, _, ang_vel = self.kinematic_model(
            self.x, self.y, self.yaw, self.v, self.throttle, self.delta
        )

        # print(f"Cross-track term: {self.crosstrack_error}{' '*10}", end="\r")

        self.logs["velocity"].append(self.v * self.dt)
        self.logs["angVel"].append(ang_vel * self.dt)
        self.logs["trueCarPos"].append(tuple((self.x, self.y)))
