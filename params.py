class RobotParams:
    def __init__(self):
        self.num_joints = None
        self.params = None
        self.dh_params = None
        self.axis_orientations = None
        self.link_mass = None
        self.center_of_mass = None
        self.inertia_matrix = None

    def load_params(self, params: dict):
        self.params = params
        self.num_joints = len(params["dh"]['a'])
        self.dh_params = params["dh"]
        self.link_mass = params["link_mass"]
        self.center_of_mass = params["center_of_mass"]
        self.inertia_matrix = params["inertia_matrix"]

    def __repr__(self):
        return f"Joints: {self.num_joints}"


def get_ur3_params() -> dict:
    params = {"Manufacturer": "Universal Robots",
              "model": "UR3",
              "dh": {"a": [0.0000, -0.6370, -0.5037, 0.0000, 0.0000, 0.0000],
                     "d": [0.2363, 0.0000, 0.0000, 0.2010, 0.1593, 0.1543],
                     "alpha": [0, 0, 0, 0, 0, 0],
                     "theta": [0, 0, 0, 0, 0, 0],
                     },
              "joint_types": ["rotation", "rotation", "rotation", "rotation", "rotation", "rotation"],
              "joint_ranges": [[-360, 360], [-360, 360], [-360, 360], [-360, 360], [-360, 360], [-360, 360]],
              "link_mass": [16.343, 28.542, 7.156, 3.054, 3.126, 0.926],
              "center_of_mass": {"Link 0": [-0.0001, -0.0600, 0.0069],
                                 "Link 1": [0.3894, 0.0000, 0.2103],
                                 "Link 2": [0.2257, 0.0007, 0.0629],
                                 "Link 3": [0.0000, -0.0048, 0.0353],
                                 "Link 4": [0.0000, 0.0046, 0.0341],
                                 "Link 5": [0.0000, 0.0000, -0.0293]},
              "inertia_matrix": {"link 0": [0.0883, -0.0001, -0.0001, -0.0001, 0.0764, 0.0076, -0.0001, 0.0076, 0.0830],
                                 "link 1": [0.1379, 0.0001, -0.0451, 0.0001, 2.5013, -0.0000, -0.0451, -0.0000, 2.4751],
                                 "link 2": [0.0236, 0.0000, -0.0168, 0.0009, 0.3388, 0.0001, -0.0168, 0.0001, 0.3353],
                                 "link 3":  [0.0056, 0.0000, 0.0000, 0.0000, 0.0051, 0.0006, 0.0000, 0.0006, 0.0043],
                                 "link 4": [0.0060, 0.0000, 0.0000, 0.0000, 0.0056, -0.0006, 0.0000, -0.0006, 0.0046],
                                 "link 5": [0.0009, 0.0000, 0.0000, 0.0000, 0.0009, 0.0000, 0.0000, 0.0000, 0.0012]}
              }
    return params


def get_scara_params() -> dict:
    params = {"Manufacturer": "Placeholder",
              "model": "Scara",
              "dh": {"a": [0.0000, 420, 250, 0.0000],
                     "d": [51.49+12, 0.0000, 0.0000, -12],
                     "alpha": [0, 0, 0, 90],
                     "theta": [0, 0, 0, 0]
                     },
              "joint_types": ["translation", "rotation", "rotation", "rotation"],
              "joint_ranges": [[], [-360, 360], [-360, 360], [-360, 360]],
              "link_mass": [],
              "center_of_mass": {"Link 0": [],
                                 "Link 1": [],
                                 "Link 2": [],
                                 "Link 3": []
                                 },
              "inertia_matrix": {"link 0": [],
                                 "link 1": [],
                                 "link 2": [],
                                 "link 3": []
                                 }
              }
    return params
