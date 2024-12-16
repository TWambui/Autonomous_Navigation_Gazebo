#! /usr/bin/env python3

import rospy 
from geometry_msgs.msg import Twist 
from nav_msgs.msg import Odometry 
from gazebo_msgs.msg import ModelStates
import math 
import numpy as np 
import tf 
from collections import deque
# import cv_bridge
import ros_numpy
from sensor_msgs.msg import Image
from ultralytics import YOLO
import torch
import cv2
from torchvision import transforms
from model import CNN_LSTM

class SideWalkDetection:
    def __init__(self) -> None:
        rospy.init_node("Seg_Node")
        rospy.loginfo("Segmentation Node Activated")
        
    
        #prepare the velocity model
        self.sequence_buffer = deque(maxlen=10)
        self.transform = transforms.ToTensor()
        self.vel_model = CNN_LSTM()
        model_state_dict = torch.load("/home/tanner/catkin_ws/src/autonomous_navigation/src/CNN_LSTM-0.00836228-0000.pth")
        self.vel_model.load_state_dict(model_state_dict)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vel_model.eval()
        self.vel_model.to(self.device)

        #prepare subscriptions and publishers
        self.image_sub = rospy.Subscriber("/image_raw", Image, self.get_image_callback)
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10) 
        self.seg_image_pub = rospy.Publisher("/segmented_image", Image, queue_size=15)
        self.position_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.get_position_callback)
        rospy.sleep(2)
        self.rate = rospy.Rate(10)
        
        self.motion()
    
    def get_position_callback(self, message: ModelStates):
        self.position = message.pose[-1].position
        return self
    
    def prepare_data(self, mask: np):
        feature = cv2.resize(mask, (640, 640), interpolation = cv2.INTER_LINEAR)
        feature = feature / 255.0
        feature = np.expand_dims(feature, axis=0)
        
        return feature

    def get_image_callback(self, message: Image):
        #returns the image
        self.image = ros_numpy.numpify(message)
        self.segment_image() #prepare segmentation data
        
        return self
    
    def segment_image(self):
        #image graying and thresholding
        grey_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret,self.mask = cv2.threshold(grey_image,27,255,cv2.THRESH_BINARY)
        self.sequence_buffer.append(self.transform(self.prepare_data(self.mask))) # add data to buffer 

        #to publish segmented image
        mask_image = np.zeros_like(self.image)
        mask_image[self.mask == 0] = [255, 0, 0]
        image_rgb = cv2.addWeighted(self.image, 1.0, mask_image, 2.0, 0)
       
        ros_image = ros_numpy.msgify(Image, image_rgb, encoding='bgr8')
        self.seg_image_pub.publish(ros_image)
        return self
        
    def motion(self):
        while self.position.x != 20.0 and self.position.y != 0: #target location
            self.rate.sleep()
            if len(self.sequence_buffer) == self.sequence_buffer.maxlen:
                feature = torch.stack(list(self.sequence_buffer)).permute(0,2,1,3)    
                feature = feature.unsqueeze(0)
                out = self.vel_model(feature.to(torch.float32).to(self.device))
                w = out.squeeze().detach().cpu().numpy()
                rospy.loginfo(w[-1])
                self.publish_vel(0.5, w[-1])
        

    def publish_vel(self, v, w):
        # using the segmented mask, predict the angular velocity
        
        cmd = Twist() 
        cmd.angular.z = w 
        cmd.linear.x = v 

        self.vel_pub.publish(cmd)

if __name__ == "__main__":
    sidewalk_navigation = SideWalkDetection()
    rospy.spin()
