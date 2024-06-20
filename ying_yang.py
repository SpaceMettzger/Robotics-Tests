from fundamentals import *
import matplotlib.pyplot as plt
import numpy as np


class YingYang:
    def __init__(self, radius, point_1):
        self.radius = radius
        self.points = {"P0": point_1}
        self.calculate_points()
        self.semicircles = self.get_semicircles()

    def calculate_points(self):
        p_0 = self.points["P0"]
        radius = self.radius

        offsets = [
            (-radius, -radius, 0),
            (0, -2 * radius, 0),
            (radius, -radius, 0),
            (-radius / 2, -radius / 2, 0),
            (0, -radius, 0),
            (radius / 2, -1.5 * radius, 0),
            (0, -radius / 4, 0),
            (-radius / 4, -radius / 2, 0),
            (0, -0.75 * radius, 0),
            (radius / 4, -radius / 2, 0),
            (0, -(radius + radius / 4), 0),
            (-radius / 4, -(radius + 0.5 * radius), 0),
            (0, -(radius + 0.75 * radius), 0),
            (radius / 4, -(radius + 0.5 * radius), 0)
        ]

        for i, (dx, dy, dz) in enumerate(offsets):
            point_name = f"P{i + 1}"
            new_point = Point(p_0.x + dx, p_0.y + dy, p_0.z + dz)
            self.points[point_name] = new_point

    def get_semicircles(self):
        return [
            [self.points["P0"], self.points["P1"], self.points["P2"]],
            [self.points["P2"], self.points["P3"], self.points["P0"]],
            [self.points["P0"], self.points["P4"], self.points["P5"]],
            [self.points["P5"], self.points["P6"], self.points["P2"]],
            [self.points["P7"], self.points["P8"], self.points["P9"]],
            [self.points["P9"], self.points["P10"], self.points["P7"]],
            [self.points["P11"], self.points["P12"], self.points["P13"]],
            [self.points["P13"], self.points["P14"], self.points["P11"]]
        ]

    def print_movement_commands(self):
        print("\nSemicircle without angled plane:")
        for i in range(0, 7):
            semicircle = self.semicircles[i]
            point_1, point_2 = semicircle[1], semicircle[2]
            print(f"ZR X {point_1.x} Y {point_1.y} Z {point_1.z} X {point_2.x} Y {point_2.y} Z {point_2.z} "
                  f"A {point_2.a} B {point_2.b} C{point_2.c}")

        print("\nSemicircle with angled plane:")
        for i in range(0, 8):
            semicircle = self.semicircles[i]
            point_1, point_2 = semicircle[1], semicircle[2]
            print(f"ZR X {point_1.x_transformed} Y {point_1.y_transformed} Z {point_1.z_transformed} "
                  f"X {point_2.x_transformed} Y {point_2.y_transformed} Z {point_2.z_transformed} "
                  f"A {point_2.a} B {point_2.b} C{point_2.c}")

    def plot_yingyang(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for triplet in self.get_semicircles():
            PointPlotter.plot_semicircle(ax, triplet[0], triplet[1], triplet[2], "regular")

            PointPlotter.plot_semicircle(ax, triplet[0], triplet[1], triplet[2], "transformed")

        for point in self.points.values():
            ax.scatter(point.x, point.y, point.z, color='r')
            ax.scatter(point.x_transformed, point.y_transformed, point.z_transformed, color='b')

        plt.show()


if __name__ == "__main__":
    delta_y = 0.2 - (-382.5)
    delta_z = -37.85 - (-14.46)

    alpha = math.atan(delta_z / delta_y)
    print(alpha / np.pi * 180)

    rot_matrix = Transformations.calculate_rotation_matrix_x(alpha)

    ying_yang = YingYang(50, Point(500, -50, -34.6))

    list(map(lambda point: point.transform_coords(rot_matrix), ying_yang.points.values()))
    for point in list(ying_yang.points.values()):
        print(point)

    ying_yang.print_movement_commands()
    ying_yang.plot_yingyang()
