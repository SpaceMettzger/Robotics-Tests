import numpy as np
import math
import matplotlib.pyplot as plt


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


class PointPlotter:
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


class Joint:
    JOINT_ID = 0

    def __init__(self,
                 joint_length: float  = 0,
                 joint_height: float = 0,
                 joint_width: float = 0,
                 current_angle: float = 0,
                 next_orientation_axis: str = "z",
                 translation_axis: str = "x",
                 rotation_range: list = (-360, 360),
                 joint_type: str = "rotation"):
        self.previous_joint = None
        self.next_joint = None
        self.joint_x_distance = joint_length
        self.joint_z_distance = joint_height
        self.joint_y_distance = joint_width
        self.joint_type = joint_type
        self._check_joint_type(joint_type)
        self.translation_axis = translation_axis
        self._check_translation_axis(translation_axis)
        self.next_joint_orientation_axis = next_orientation_axis
        self._check_rotation_axis(next_orientation_axis)
        self.current_angle = current_angle
        self.next_joint_orientation = self.get_orientation_matrix()
        self.rotation_range_lower_bounds = rotation_range[0]
        self.rotation_range_upper_bounds = rotation_range[1]
        self.joint_id = Joint.JOINT_ID
        Joint.JOINT_ID += 1

    def _check_joint_type(self, joint_type):
        if joint_type not in ["rotation", "translation"]:
            print(f"Invalid joint type: {joint_type}. Defaulting to rotational joint")
            self.joint_type = "rotation"

    def _check_translation_axis(self, translation_axis):
        if translation_axis not in ["x", "y", "z"]:
            print(f"Invalid translation axis: {translation_axis}. Defaulting to 'z' as translation axis")
            self.translation_axis = "z"

    def _check_rotation_axis(self, rotation_axis):
        if rotation_axis not in ["x", "y", "z"]:
            print(f"Invalid rotation axis: {rotation_axis}. Defaulting to 'z' as rotation axis")
            self.rotation_axis = "z"

    def get_orientation_matrix(self, base: bool = False, angle: float = 0):
        angle = angle / math.pi * 180
        if not self.previous_joint:
            return Transformations.get_base_coords()
        if base or self.next_joint_orientation_axis.lower() == "z":
            return Transformations.calculate_rotation_matrix_z(angle)
        elif self.next_joint_orientation_axis.lower() == "x":
            return Transformations.calculate_rotation_matrix_x(angle)
        elif self.next_joint_orientation_axis.lower() == "y":
            return Transformations.calculate_rotation_matrix_y(angle)
        else:
            print("Orientation axis variable is invalid")

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
            # If the angle exceeds +-360 degrees, it will be limited to +-360:
            angle = ((angle + 360) % 720) - 360
            print(f"Setting new Angle: {angle} for joint {self.joint_id}")
            # joint.next_joint_orientation = joint.get_orientation_matrix(base=True, angle=angle)
            joint.current_angle = angle
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
        if self.translation_axis == "x":
            joint.joint_x_distance = translation_value
        elif self.translation_axis == "y":
            joint.joint_y_distance = translation_value
        elif self.translation_axis == "z":
            joint.joint_z_distance = translation_value

    def get_tcp_position(self, joints: list):
        if len(joints) != Joint.JOINT_ID:
            print(f"Provided parameter is not equal to number of joints. "
                  f"Number of needed parameters is {Joint.JOINT_ID}")
            return
        base_joint = self
        while base_joint.previous_joint is not None:
            base_joint = base_joint.previous_joint

        position = np.matrix(np.eye(4))
        transformations = [position]

        current_joint = base_joint
        for joint_index in range(len(joints)):
            if current_joint is None:
                raise ValueError("Number of joints exceeds the provided joint parameters.")
            current_joint.current_angle = joints[joint_index]

            # rotation_1 = current_joint.get_orientation_matrix(angle=current_joint.current_angle)
            translation = Transformations.get_translation_matrix(current_joint.joint_z_distance,
                                                                 current_joint.joint_x_distance)

            if current_joint.joint_type == "translation":
                rotation = current_joint.get_orientation_matrix(base=True, angle=0)
            else:
                rotation = current_joint.get_orientation_matrix(base=False, angle=joints[joint_index])
            transformation = np.matmul(rotation, translation)

            transformations.append(transformation)
            current_joint = current_joint.next_joint

        position = np.eye(4)
        for index, transformation in enumerate(transformations):
            position = np.matmul(position, transformation)
            transformations[index] = position

        print("TCP:")
        print(position)

        return position, transformations

    def plot_forwards_kinematic(self, angles: list):
        _, transformations = self.get_tcp_position(angles)

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

        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
        # ax.set_zlim(axis_limits)

        ax.legend()
        plt.show()

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
    def get_translation_matrix(heigth, length):
        return np.matrix([
            [1, 0, 0, length],
            [0, 1, 0, 0],
            [0, 0, 1, heigth],
            [0, 0, 0, 1]
        ])
