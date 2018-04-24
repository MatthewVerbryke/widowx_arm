#!/usr/bin/env python

"""
  'workspace.py' -
  A crude, quickly-made robot workspace analysis script using MoveIt's
  FK solvers.
  
  BSD 3-Clause License
  
  Copyright (c) 2018, University of Cincinnati
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  
  TODO: improve documentation across the board
"""


import csv
import os
import sys
import time

import numpy as np
from geometry_msgs.msg import PoseStamped
import moveit_commander
import moveit_msgs.msg
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest, GetPositionFKResponse
import rospy
from operator import mul


#TODO: PASS THESE IN AS ARGUMENTS
ARM_GROUP = "widowx_arm"
GRIPPER_GROUP = "widowx_gripper"
REF_FRAME = "base_link"
LAUNCH_RVIZ = False
WIDOWX_JOINT_LIMITS =  [[-2.617, 2.167],
                        [-1.571, 1.571],
                        [-1.571, 1.571],
                        [-1.745, 1.745],
                        [-2.617, 2.617]]


class MoveItInterface(object):
    """ 
    An application-specific, 'lite' MoveIt! interface object.
    """
    
    def __init__(self):
        """
        Initialize the move group interface for the selected planning 
        group.
        """
        
        # Get the planning group and reference frame
        # TODO: PASS IN THROUGH ARGUMENTATION
        self.planning_group = ARM_GROUP
        gripper_group = GRIPPER_GROUP
        self.ref_frame = REF_FRAME
        
        # Initialize moveit_commander
        moveit_commander.roscpp_initialize(sys.argv)
        
        # Initialize rospy node
        rospy.init_node("{}_moveit_interface".format(self.planning_group), anonymous=True)
        
        # Initialize cleanup for this node
        rospy.on_shutdown(self.cleanup)
        
        # Wait for RViz to start up, if we are running it
        if LAUNCH_RVIZ:
            rospy.sleep(10)
            
        # Initialize a move group for the main arm planning group
        self.arm = moveit_commander.MoveGroupCommander("{}".format(self.planning_group))
        
        # Initialize a move group for the gripper planning group
        self.gripper = moveit_commander.MoveGroupCommander("{}".format(gripper_group))
        
        # Get end effector link name
        self.eef_link = self.arm.get_end_effector_link()
        
        # Set the pose target reference frame
        self.target_ref_frame = self.ref_frame
        self.arm.set_pose_reference_frame(self.ref_frame)
        
        # Add tolerence to goal position and orientation
        self.arm.set_goal_position_tolerance(0.1)
        self.arm.set_goal_orientation_tolerance(0.1)
        rospy.loginfo("MoveIt! interface initialized...")
        
        # Setup a connection to the 'compute_fk' service
        self.fk_srv = rospy.ServiceProxy("/compute_fk", GetPositionFK)
        self.fk_srv.wait_for_service()
        rospy.loginfo("FK service initialized...")

    def fk_solve(self, joint_angles, link_names):
        """
        Given a set of joint angles for the robot, use forward kinematics
        to determine the end-effector pose reached with those angles
        https://github.com/uts-magic-lab/moveit_python_tools/blob/master/src/moveit_python_tools/get_fk.py
        """
        
        # Build the service request
        req = GetPositionFKRequest()
        req.header.frame_id = self.ref_frame
        req.fk_link_names = [link_names]
        req.robot_state.joint_state.name = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
        req.robot_state.joint_state.position = joint_angles
        
        # Try to send the request to the 'compute_fk' service
        try:
            resp = self.fk_srv.call(req)
            return resp
        except rospy.ServiceException:
            rospy.logerr("Service execption:" + str(rospy.ServiceException))
            
    def cleanup(self):
        """
        Things to do when shutdown occurs.
        """

        # Shut down MoveIt cleanly
        moveit_commander.roscpp_shutdown()
        
        # Exit MoveIt
        moveit_commander.os._exit(0)
        
        # Log shutdown
        rospy.loginfo("Shutting down node '{}_moveit_interface'".format(self.planning_group))
        rospy.sleep(1)


def generate_robot_workspace(joint_limits):
    """
    Use forward kinematics to determine the workspace of a robotic arm.
    """
    
    # Create new MoveIt interface object
    interface = MoveItInterface()
    
    angle_range = []
    range_len = []
    
    # Determine angle range to run through based on the joint limits specified
    for i in range(0,len(joint_limits)):
        low = joint_limits[i][0]
        high = joint_limits[i][1]
        arange = np.linspace(low, high, 10).tolist()
        range_len.append(len(arange))
        angle_range.append(arange)
    
    # Prepare loop
    combination_len = reduce(mul, range_len)
    eef_poses = []
    it = 0
    
    # A giant mess of nested for-loops to send each combination out to the fk solver
    for i in range(0,range_len[0]):
        for j in range(0,range_len[1]):
            for k in range(0,range_len[2]):
                for l in range(0,range_len[3]):
                    for m in range(0,range_len[4]):
                        
                        # Construct the combination
                        joint_angles = [angle_range[0][i],
                                        angle_range[1][j], 
                                        angle_range[2][k],
                                        angle_range[3][l],
                                        angle_range[4][m]]
                                        
                        # Send it to the fk solver and get a response
                        resp = interface.fk_solve(joint_angles, "wrist_2_link")

                        # Check if result is good (i.e. no error codes) and store the result
                        if (resp.error_code.val == 1):
                            point = [resp.pose_stamped[0].pose.position.x,
                                     resp.pose_stamped[0].pose.position.y, 
                                     resp.pose_stamped[0].pose.position.z]
                            eef_poses.append(point)
                            # Display the progress of the run
                            it += 1
                            sys.stdout.write("\r" + "Progress: {}%".format(float(it)/float(combination_len)*100.0))
                            sys.stdout.flush()
                        else:
                            print(" ")
                            rospy.logerr("Error code {} returned! Shutting down.".format(resp.error_code))
                            exit()
                        

                        
    # Send it to a csv file for later analysis
    with open("left_widowx_ws.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(eef_poses)
        print(" ")


if __name__ == "__main__":
    generate_robot_workspace(WIDOWX_JOINT_LIMITS)
