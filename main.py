from fundamentals import *
from ying_yang import YingYang
import numpy as np


if __name__ == "__main__":
    delta_y = 0.2 - (-382.5)
    delta_z = -37.85 - (-14.46)

    alpha = math.atan(delta_z/delta_y)
    print(alpha / np.pi * 180)

    rot_matrix = Rotations.calculate_rotation_matrix_x(alpha)

    ying_yang = YingYang(50, Point(500, -50, -34.6))

    list(map(lambda point: point.transform_coords(rot_matrix), ying_yang.points.values()))
    for point in list(ying_yang.points.values()):
        print(point)


    # PointPlotter(list(ying_yang.points.values())).plot_transformed_points(rot_matrix, ying_yang.semicircles)
    PointPlotter(list(ying_yang.points.values())).plot_yingyang(ying_yang)