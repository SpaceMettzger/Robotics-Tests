from fundamentals import *


if __name__ == "__main__":
    chained_joints = Joint(
        joint_length=0, joint_height=51.49, translation_axis="z", joint_type="translation")
    chained_joints.add_joint(Joint(
        joint_length=420, next_orientation_axis="z", rotation_range=[-360, 360]))
    chained_joints.add_joint(Joint(
        joint_length=250, next_orientation_axis="z", rotation_range=[-360, 360]))
    chained_joints.add_joint(Joint(
        joint_height=-12, next_orientation_axis="z", rotation_range=[-360, 360]))

    # chained_joints.change_joint_translation(510.49, 0)
    chained_joints.change_joint_angle(11.34, 1)
    chained_joints.change_joint_angle(55.80, 2)
    chained_joints.change_joint_angle(-190.31, 3)

    angles = [0, 11.34, 55.80, -190.31]
    chained_joints.plot_forwards_kinematic(angles)