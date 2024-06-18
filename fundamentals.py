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
