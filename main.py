from fundamentals import *


if __name__ == "__main__":
    joint_chain = Joint(
        joint_length=0, joint_height=51.49+12, translation_axis="z", joint_type="translation")
    joint_chain.add_joint(Joint(
        joint_length=0, next_orientation_axis="z", rotation_range=[-360, 360]))
    joint_chain.add_joint(Joint(
        joint_length=420, next_orientation_axis="z", rotation_range=[-360, 360]))
    joint_chain.add_joint(Joint(
        joint_length=250, next_orientation_axis="z", rotation_range=[-360, 360]))
    joint_chain.add_joint(Joint(
        joint_height=-12, next_orientation_axis="z", rotation_range=[-360, 360]))

    joint_chain.change_joint_translation(51.49+12, 0)
    # joint_chain.change_joint_angle(11.34, 1)
    joint_chain.change_joint_angle(11.34, 2)
    joint_chain.change_joint_angle(55.8, 3)
    joint_chain.change_joint_angle(-190.31, 4)

    # angles = [0, 108, 90, 90, 90]
    Plotter.plot_forwards_kinematic(joint_chain)
