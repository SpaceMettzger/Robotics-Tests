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
        self.x_transformed = self.vector_transformed[0]
        self.y_transformed = self.vector_transformed[1]
        self.z_transformed = self.vector_transformed[2]

    def __repr__(self):
        return f"{self.point_name}, {self.vector[:3]}"


class PointPlotter:
    def __init__(self, points: list):
        self.points = points

    def plot_transformed_points(self, rot_matrix, semicircles):
        original_x = []
        original_y = []
        original_z = []
        transformed_x = []
        transformed_y = []
        transformed_z = []

        for point in self.points:
            transformed_vector = np.dot(rot_matrix, point.vector)
            transformed_point = transformed_vector.A1[:3]

            original_x.append(point.x)
            original_y.append(point.y)
            original_z.append(point.z)

            transformed_x.append(transformed_point[0])
            transformed_y.append(transformed_point[1])
            transformed_z.append(transformed_point[2])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(original_x, original_y, original_z, c='blue', marker='o', label='Original Points')
        ax.scatter(transformed_x, transformed_y, transformed_z, c='red', marker='^', label='Transformed Points')

        # Add semicircles for Yin and Yang
        for semicircle in semicircles:
            self._plot_semicircle(ax, semicircle, rot_matrix)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('3D Rotation of Points with Yin-Yang Semicircles')
        ax.set_aspect('equal', adjustable='box')

        plt.show()

    def _plot_semicircle(self, ax, points_triplet, rot_matrix):
        p_start, p_mid, p_end = points_triplet

        # Generate points for the semicircle
        t = np.linspace(0, 1, 100)
        x_semi = p_start
        y_semi = (1 - t) ** 2 * p_start.y + 2 * (1 - t) * t * p_mid.y + t ** 2 * p_end.y
        z_semi = (1 - t) ** 2 * p_start.z + 2 * (1 - t) * t * p_mid.z + t ** 2 * p_end.z

        # Transform semicircle points
        transformed_semi = [
            np.dot(rot_matrix, np.array([x, y, z, 1])).A1[:3]
            for x, y, z in zip(x_semi, y_semi, z_semi)
        ]

        # Extract transformed x, y, z for plotting
        transformed_x_semi, transformed_y_semi, transformed_z_semi = zip(*transformed_semi)

        ax.plot(transformed_x_semi, transformed_y_semi, transformed_z_semi, label='Semicircle')

    def plot_semicircle(self, ax, p1, p2, p3):
        # Convert Points to numpy arrays
        P1 = np.array([p1.x, p1.y, p1.z])
        P2 = np.array([p2.x, p2.y, p2.z])
        P3 = np.array([p3.x, p3.y, p3.z])

        # Calculate the midpoint
        M = (P1 + P3) / 2

        # Calculate the radius
        radius = np.linalg.norm(P1 - M)

        # Calculate the plane's normal vector
        v1 = P2 - P1
        v2 = P3 - P1
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        # Calculate the perpendicular vector
        P1P3_unit = (P3 - P1) / np.linalg.norm(P3 - P1)
        perp_vector = np.cross(normal, P1P3_unit)

        # Parametrize the semicircle
        angles = np.linspace(0, np.pi, 100)
        semicircle_points = []
        for angle in angles:
            point_on_circle = (
                    M + radius * np.cos(angle) * P1P3_unit + radius * np.sin(angle) * perp_vector
            )
            semicircle_points.append(point_on_circle)

        semicircle_points = np.array(semicircle_points)
        ax.plot(semicircle_points[:, 0], semicircle_points[:, 1], semicircle_points[:, 2])

    def plot_semicircle_transformed(self, ax, p1, p2, p3):
        # Convert Points to numpy arrays
        P1 = np.array([p1.x_transformed, p1.y_transformed, p1.z_transformed])
        P2 = np.array([p2.x_transformed, p2.y_transformed, p2.z_transformed])
        P3 = np.array([p3.x_transformed, p3.y_transformed, p3.z_transformed])

        # Calculate the midpoint
        M = (P1 + P3) / 2

        # Calculate the radius
        radius = np.linalg.norm(P1 - M)

        # Calculate the plane's normal vector
        v1 = P2 - P1
        v2 = P3 - P1
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        # Calculate the perpendicular vector
        P1P3_unit = (P3 - P1) / np.linalg.norm(P3 - P1)
        perp_vector = np.cross(normal, P1P3_unit)

        # Parametrize the semicircle
        angles = np.linspace(0, np.pi, 100)
        semicircle_points = []
        for angle in angles:
            point_on_circle = (
                    M + radius * np.cos(angle) * P1P3_unit + radius * np.sin(angle) * perp_vector
            )
            semicircle_points.append(point_on_circle)

        semicircle_points = np.array(semicircle_points)
        ax.plot(semicircle_points[:, 0], semicircle_points[:, 1], semicircle_points[:, 2])

    def plot_yingyang(self, yingyang):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for triplet in yingyang.get_semicircles():
            self.plot_semicircle(ax, triplet[0], triplet[1], triplet[2])

            self.plot_semicircle_transformed(ax, triplet[0], triplet[1], triplet[2])

        # Plot all points for reference
        for point in yingyang.points.values():
            ax.scatter(point.x, point.y, point.z, color='r')
            ax.scatter(point.x_transformed, point.y_transformed, point.z_transformed, color='b')

        plt.show()




class Joint:
    def __init__(self, joint_length: float, orientation: np.matrix):
        self.joint_length = joint_length
        self.orientation = orientation
        self.previous_joint = None
        self.next_joint = None

    def add_joint(self, next_joint):
        self.next_joint = next_joint
        next_joint.previous_joint = self


class Rotations:

    @staticmethod
    def calculate_rotation_matrix_x(alpha):
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        rot_matrix_x = np.matrix([
            [1, 0, 0, 0],
            [0, cos_a, -sin_a, 0],
            [0, sin_a, cos_a, 0],
            [0, 0, 0, 1]
        ])
        return rot_matrix_x
