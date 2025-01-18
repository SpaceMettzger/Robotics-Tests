import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from params import *


class Point:
    POINT_INDEX = 0

    def __init__(self, x: float, y: float, z: float, a: float = 135, b: float = 0, c: float = 0):
        self.point_name = "P" + str(Point.POINT_INDEX)
        Point.POINT_INDEX = Point.POINT_INDEX + 1 if Point.POINT_INDEX < 14 else 1
        self.x = x
        self.y = y
        self.z = z
        self.x_transformed = None
        self.y_transformed = None
        self.z_transformed = None
        self.a = a
        self.b = b
        self.c = c
        self.vector = self.create_array()
        self.vector_transformed = None

    def create_array(self):
        return np.array([self.x, self.y, self.z, 1])

    def transform_coords(self, rotation_matrix: np.matrix):
        self.vector_transformed = np.dot(rotation_matrix, self.vector).A1[:3]
        self.x_transformed = round(self.vector_transformed[0], 3)
        self.y_transformed = round(self.vector_transformed[1], 3)
        self.z_transformed = round(self.vector_transformed[2], 3)

    def __repr__(self):
        return f"{self.point_name}, {self.vector[:3]}"


class Plotter:
    def __init__(self, points: list):
        self.points = points

    @staticmethod
    def plot_semicircle(ax, point_1, point_2, point_3, orientation: str = "regular"):
        if orientation.lower() == "regular":
            p1 = np.array([point_1.x, point_1.y, point_1.z])
            p2 = np.array([point_2.x, point_2.y, point_2.z])
            p3 = np.array([point_3.x, point_3.y, point_3.z])

        elif orientation.lower() == "transformed":
            p1 = np.array([point_1.x_transformed, point_1.y_transformed, point_1.z_transformed])
            p2 = np.array([point_2.x_transformed, point_2.y_transformed, point_2.z_transformed])
            p3 = np.array([point_3.x_transformed, point_3.y_transformed, point_3.z_transformed])

        else:
            return

        m = (p1 + p3) / 2

        radius = np.linalg.norm(p1 - m)

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        p1p3_unit = (p3 - p1) / np.linalg.norm(p3 - p1)
        perp_vector = np.cross(normal, p1p3_unit)

        angles = np.linspace(0, np.pi, 100)
        semicircle_points = []
        for angle in angles:
            point_on_circle = (
                    m + radius * np.cos(angle) * p1p3_unit + radius * np.sin(angle) * perp_vector
            )
            semicircle_points.append(point_on_circle)

        semicircle_points = np.array(semicircle_points)
        ax.plot(semicircle_points[:, 0], semicircle_points[:, 1], semicircle_points[:, 2])

    @staticmethod
    def plot_forwards_kinematic(joint_chain, angles: list = None):
        _, transformations = joint_chain.get_tcp_position(angles)

        translations = [matrix[:3, 3] for matrix in transformations]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x_coords = [t[0] for t in translations]
        y_coords = [t[1] for t in translations]
        z_coords = [t[2] for t in translations]

        ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o', label='Joints')

        ax.plot(x_coords, y_coords, z_coords, label='Robot Links')

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        all_coords = np.array([x_coords, y_coords, z_coords])
        axis_limits = [np.min(all_coords), np.max(all_coords)]

        ax.set_xlim(-axis_limits[1], axis_limits[1])
        ax.set_ylim(-axis_limits[1], axis_limits[1])
        # ax.set_zlim(axis_limits)

        for i, (x, y, z) in enumerate(zip(x_coords, y_coords, z_coords)):
            if i == 0:
                continue
            ax.text(int(x), int(y), int(z), f'J {i-1}', color='blue')

        ax.legend()
        plt.show()


class Robot:
    def __init__(self):
        self.params = None
        self.joints = None

    def set_up_joint_chain(self, params):
        if params.num_joints is None:
            print("No Parameters loaded. Aborting")
            return
        self.params = params
        for i in range(0, self.params.num_joints):
            if i == 0:
                self.joints = Joint(joint_length=self.params.dh_params["a"][i],
                                    joint_height=self.params.dh_params["d"][i],
                                    theta=self.params.dh_params["theta"][i],
                                    alpha=self.params.dh_params["alpha"][i])
            else:
                self.joints.add_joint(Joint(joint_length=self.params.dh_params["a"][i],
                                            joint_height=self.params.dh_params["d"][i],
                                            theta=self.params.dh_params["theta"][i],
                                            alpha=self.params.dh_params["alpha"][i]))

    def __repr__(self):
        return f"Robot: {self.params.params['Manufacturer']}, {self.params.params['model']}"


class Joint:
    JOINT_ID = 0

    def __init__(self,
                 joint_length: float = 0,
                 joint_height: float = 0,
                 theta: float = 0,
                 alpha: float = 0,
                 rotation_range: list = (-360, 360),
                 joint_type: str = "rotation"):
        self.previous_joint = None
        self.next_joint = None
        self.joint_x_distance = joint_length
        self.joint_z_distance = joint_height
        self.joint_type = joint_type
        self._check_joint_type(joint_type)
        self.alpha = alpha
        self.theta = theta
        self.rotation_range_lower_bounds = rotation_range[0]
        self.rotation_range_upper_bounds = rotation_range[1]
        self.joint_id = Joint.JOINT_ID
        Joint.JOINT_ID += 1

    def _check_joint_type(self, joint_type):
        if joint_type not in ["rotation", "translation"]:
            print(f"Invalid joint type: {joint_type}. Defaulting to rotational joint")
            self.joint_type = "rotation"

    def get_orientation_matrix(self, base: bool = False, angle: float = 0):
        angle = math.radians(angle)
        if not self.previous_joint:
            return Transformations.get_base_coords()
        if base:
            return Transformations.calculate_rotation_matrix_z(angle)
        else:
            return Transformations.calculate_rotation_matrix_z(angle)

    def add_joint(self, next_joint):
        current_joint = self
        while current_joint.next_joint is not None:
            current_joint = current_joint.next_joint
        current_joint.next_joint = next_joint
        next_joint.previous_joint = current_joint

    def _find_joint_by_joint_nr(self, joint_nr):
        joint = self
        while joint and joint.joint_id != joint_nr:
            if joint.joint_id < joint_nr:
                joint = joint.next_joint
            else:
                joint = joint.previous_joint
        print(joint.joint_id)
        return joint

    def change_joint_angle(self, angle: float, joint_nr: int):
        joint = self._find_joint_by_joint_nr(joint_nr)
        if joint is None:
            print(f"Joint {joint_nr} not found.")
            return
        if joint.joint_type == "translation":
            print("Can't change angle of translation joint")
            return

        if joint.rotation_range_upper_bounds >= angle >= joint.rotation_range_lower_bounds:
            angle = ((angle + 360) % 720) - 360
            print(f"Setting new Angle: {angle} for joint {joint.joint_id}")
            joint.theta = angle
            print(f"New rotation:\n{joint.get_orientation_matrix(angle=angle)}")
        else:
            print("Rotation exceeding joint limits. Rotation aborted")

    def change_joint_translation(self, translation_value: float, joint_nr: int):
        joint = self._find_joint_by_joint_nr(joint_nr)
        if joint is None:
            print(f"Joint {joint_nr} not found.")
            return
        if joint.joint_type == "rotation":
            print("Can't change translation of rotation joint")
            return
        joint.joint_height = translation_value
        print(f"Setting new Angle: {translation_value} for joint {joint.joint_id}")

    def get_tcp_position(self, angles: list = None):
        if angles and len(angles) != Joint.JOINT_ID:
            print(
                f"Provided parameter is not equal to number of joints. Number of needed parameters is {Joint.JOINT_ID}")
            return

        # Find the base joint
        base_joint = self

        while base_joint.previous_joint is not None:
            base_joint = base_joint.previous_joint

        # Initialize the position matrix
        position = np.matrix(np.eye(4))
        transformations = [position]

        # Traverse each joint and calculate the transformation matrices
        current_joint = base_joint
        for joint_index in range(Joint.JOINT_ID):

            theta = angles[joint_index] if angles else current_joint.theta

            # Construct transformation matrix based on DH parameters
            translation = Transformations.get_translation_matrix(current_joint.joint_z_distance, current_joint.joint_x_distance)
            rotation_z = Transformations.calculate_rotation_matrix_z(theta)
            rotation_x = Transformations.calculate_rotation_matrix_x(current_joint.alpha)

            # Combine the transformations
            transformation = rotation_z @ translation @ rotation_x

            # Update the overall position matrix
            position = position @ transformation
            transformations.append(position)

            # Move to the next joint
            current_joint = current_joint.next_joint

        print("TCP (Tool Center Point):")
        print(position)

        return position, transformations


# noinspection PyTypeChecker
class Transformations:

    @staticmethod
    def get_base_coords():
        base = np.matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return base

    @staticmethod
    def calculate_rotation_matrix_x(alpha):
        alpha = math.radians(alpha)
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        rot_matrix_x = np.matrix([
            [1,     0,      0,      0],
            [0, cos_a, -sin_a,      0],
            [0, sin_a,  cos_a,      0],
            [0,     0,      0,      1]
        ])
        return rot_matrix_x

    @staticmethod
    def calculate_rotation_matrix_y(alpha):
        alpha = math.radians(alpha)
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        rot_matrix_y = np.matrix([
            [cos_a,     0,  sin_a,  0],
            [0,         1,      0,  0],
            [-sin_a,    0,  cos_a,  0],
            [0,         0,      0,  1]
        ])
        return rot_matrix_y

    @staticmethod
    def calculate_rotation_matrix_z(alpha):
        alpha = math.radians(alpha)
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        rot_matrix_z = np.matrix([
            [cos_a, -sin_a,     0,      0],
            [sin_a,  cos_a,     0,      0],
            [0,          0,     1,      0],
            [0,          0,     0,      1]
        ])
        return rot_matrix_z

    @staticmethod
    def get_translation_matrix(height, length):
        return np.matrix([
            [1, 0, 0, length],
            [0, 1, 0, 0],
            [0, 0, 1, height],
            [0, 0, 0, 1]
        ])
