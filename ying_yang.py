from fundamentals import Point


class YingYang:
    def __init__(self, radius, point_1):
        self.radius = radius
        self.points = {"P0": point_1}
        self.calculate_points()
        self.semicircles = []

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

