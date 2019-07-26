import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import pyrealsense2 as rs
import os
import cv2
import numpy as np
import math
from YOLOv3 import YOLOv3
from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

import time

class Detector(object):
    def __init__(self):
        self.vdo = cv2.VideoCapture()
        self.yolo3 = YOLOv3("YOLOv3/cfg/yolo_v3.cfg","YOLOv3/yolov3.weights","YOLOv3/cfg/coco.names", is_xywh=True)
        self.deepsort = DeepSort("deep_sort/deep/checkpoint/ckpt.t7")
        self.class_names = self.yolo3.class_names
        self.write_video = True

    def open(self, video_path):
        assert os.path.isfile(video_path), "Error: path error"
        self.vdo.open(video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter("demo.avi", fourcc, 20, (self.im_width,self.im_height))
        return self.vdo.isOpened()
        
    def detect(self):
        

        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


        # Start streaming
        profile = pipeline.start(config)

        xmin, ymin, xmax, ymax = 0,0, 640, 480

        try:
            while True:
                start = time.time()
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                    
                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                ori_im = color_image
                im = ori_im[ymin:ymax, xmin:xmax, (2,1,0)] #3dim (0,1,2) --> (2,1,0) index rearrange 
                bbox_xywh, cls_conf, cls_ids = self.yolo3(im) 

                if bbox_xywh is not None:
                    mask = cls_ids==0
                    bbox_xywh = bbox_xywh[mask]
                    bbox_xywh[:,3] *= 1.2
                    cls_conf = cls_conf[mask]
                    outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:,:4]
                        identities = outputs[:,-1]
                        #ori_im = draw_bboxes(ori_im, bbox_xyxy, identities, offset=(xmin,ymin)) 

                        # Modification of draw_bboxes
                        offset =(xmin,ymin)
                        for i,box in enumerate(bbox_xyxy):
                            x1,y1,x2,y2 = [int(i) for i in box]
                            #most left up point is (0,0)
                            #x1,y1 is left up point, x2,y2 is right down point // pixel unit
                            x1 += offset[0]
                            x2 += offset[0]
                            y1 += offset[1]
                            y2 += offset[1]

                            boxed_depth = depth_image[y1:y2, x1:x2]   

                            # #get closest depth in xyxy box
                            # min_depth = np.amin(boxed_depth)
                            # min_result = np.where(boxed_depth == min_depth)
                            # listOfCordinates = list(zip(min_result[0], min_result[1]))
                            # for cord in listOfCordinates:
                            #     min_pixel = cord #only use first cordinate
                            #     break
                            # min_pixel = list(min_pixel)
                            #  #revert to pixel in original depth before sliced
                            # min_pixel[0] += y1
                            # min_pixel[1] += x1

                            # Get real Distance
                            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                            depth = boxed_depth * depth_scale
                            #real_dist,_,_,_ = cv2.mean(depth) #meters unit
                            real_dist = np.median(depth)

                            # Get real Width
                            # d434's FOV Horizontal:91.2
                            width_scale = (2*real_dist*math.tan(math.radians(91.2/2))) / 640
                            real_width = width_scale*(x2-x1)

                            # Get real Height
                            # d434's FOV Vertical:65.5
                            height_scale = (2*real_dist*math.tan(math.radians(65.5/2))) / 480
                            real_height = height_scale *(y2-y1)





                            # box text and bar
                            id = int(identities[i]) if identities is not None else 0    
                            color = COLORS_10[id%len(COLORS_10)]
                            label = '{} {}, d={:.3f} w={:.3f} h={:.3f}'.format("object", id, real_dist, real_width, real_height)
                            print(label)
                            print('pixel of top left and bottom right')
                            print('(', x1, ',', y1, ')    (', x2, ',', y2, ')')

                end = time.time()
                print("time: {}s, fps: {}".format(end-start, 1/(end-start)))

                #if self.write_video:
                #    self.output.write(ori_im)


        finally:
            # Stop streaming
            pipeline.stop()
                        
         
if __name__=="__main__":
    det = Detector()
    #det.open(sys.argv[1]) #opens video  // not used for realsense
    det.detect()  #frame by frame detect











