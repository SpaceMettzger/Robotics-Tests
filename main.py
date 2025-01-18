import sys

from fundamentals import *
from params import *

if __name__ == "__main__":



    joint_chain = Joint(
        joint_length=0, joint_height=51.49+12, alpha=0, theta=0, joint_type="translation")
    joint_chain.add_joint(Joint(
        joint_length=420, alpha=0, theta=0, rotation_range=[-360, 360]))
    joint_chain.add_joint(Joint(
        joint_length=250, alpha=0, theta=0, rotation_range=[-360, 360]))
    joint_chain.add_joint(Joint(
        joint_height=-12, alpha=90, theta=0, rotation_range=[-360, 360]))

    #joint_chain.change_joint_translation(51.49+12, 0)
    joint_chain.change_joint_angle(11.34, 1)
    joint_chain.change_joint_angle(55.8, 2)
    joint_chain.change_joint_angle(-190.31, 3)

    angles = [0, 11.34, 55.8, -190.31]
    Plotter.plot_forwards_kinematic(joint_chain, angles)

    # Todo: Translation joints don't work right yet
