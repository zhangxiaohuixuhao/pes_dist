# -*- coding: utf-8 -*-  

"""
By Guanghao, based on Zhang Hui's Code, 08-JAN-2020
For YuanTou Park Pedestrian Counting Project (YZ-PARK-20191210)
"""
import os
import numpy as np
import sys
sys.path.append('.')
try:
    from queue import Queue
except Exception:
    from Queue import *

import requests
import json
import collections
import time
import argparse
import ast
import datetime
import threading
import subprocess as sp
import traceback
from xml.dom import minidom
import cv2
# from darknet_v3rt import *
import os
import pycuda.autoinit
import pycuda.driver as cuda
from utils.yolov3 import TrtYOLOv3
import math

global unnormal_send #不正常距离对数
global img_send #图像
global camera_sta #摄像头状态
global dist #最小距离
global dist_time, interval #发送时间间隔，报警间隔距离
global put_img

dist_time = 10
interval = 1
unnormal_send = 0
camera_sta = False

class RQueue(object):
    def __init__(self, max_len, name, ):
        """
        Restricted Queue, 受限队列，防止图像数据过多导致内存占满。
        By Guanghao
        :param max_len: 队列最大元素个数, 超出后则先丢掉最早元素，腾出一个空间
        :param name: 队列名称
        """
        self._q = Queue()
        self.max_len = max_len
        self.name = name

    def empty(self):
        return self._q.empty()

    def put(self, obj):
        if self._q.qsize() > self.max_len:
            self._q.get()
        return self._q.put(obj)

    def get(self):
        """
        Queue 本身实现了多线程和线程锁，如果队列是空，那么get()方法会阻塞式等待
        :return:
        """
        return self._q.get()

def warn_or_normal(ori_person, other_person):
    temp_end = []
    p0 = np.array([ori_person[2], ori_person[3]])
    for temp in other_person:
        p1 = np.array([temp[2], temp[3]])
        p2 = p1 - p0
        line_long = math.hypot(p2[0], p2[1])
        line_temp = line_long * 2 * 1.7 / ((ori_person[3] - ori_person[1]) + (temp[3] - temp[1]))
        temp_end.append(line_temp)
    return temp_end

def drawimg(ori_imgs, tempAA, center_person, interval):
    # print(interval)
    normal_num = 0
    unnormal_num = 0
    distence = 1000
    for m in tempAA: 
        ori_person = center_person[int(m[0])]
        other_person = center_person[int(m[1])]
        x_ori = ori_person[0]
        y_ori = int((ori_person[3] - ori_person[1])/2 + ori_person[1])
        x_oth = other_person[0]
        y_oth = int((other_person[3] - other_person[1])/2 + other_person[1])
        center_x = int((x_oth + x_ori) / 2)
        center_y = int((y_oth + y_ori) / 2)
        temp = m[2]
        if float(temp) > float(interval):
            cv2.circle(ori_imgs, (x_ori, y_ori), 12, (255, 220, 0), -1)
            cv2.circle(ori_imgs, (x_oth, y_oth), 12, (255, 220, 0), -1)
            cv2.line(ori_imgs,(x_ori, y_ori),(x_oth, y_oth), (255, 220, 0),4)
            cv2.putText(ori_imgs, str(round(temp, 1)), (center_x, center_y), cv2.FONT_HERSHEY_COMPLEX, 1.25, (255, 220, 0), 2)
        else:
            cv2.circle(ori_imgs, (x_ori, y_ori), 12, (100, 50, 255), -1)
            cv2.circle(ori_imgs, (x_oth, y_oth), 12, (100, 50, 255), -1)
            cv2.line(ori_imgs,(x_ori, y_ori),(x_oth, y_oth), (100, 50, 255),4)
            cv2.putText(ori_imgs, str(round(temp, 1)), (center_x, center_y), cv2.FONT_HERSHEY_COMPLEX, 1.25, (100, 50, 255), 2)
            unnormal_num = unnormal_num + 1
    distence = tempAA[len(tempAA) - 1][2]
    if unnormal_num == 0:
        distence = 0
    # print(interval, unnormal_num, distence)
    cv2.rectangle(ori_imgs, (10, 20), (550, 100), (0, 255, 255), thickness=-1)
    # cv2.putText(ori_imgs, 'Dist >  1m: ' + str(normal_num), (20, 80), cv2.FONT_HERSHEY_COMPLEX,  1.75, (255, 220, 0), 3)
    cv2.putText(ori_imgs, 'Dist <= ' + str(interval) + 'm: ' + str(unnormal_num), (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1.75, (100, 50, 255), 3)
    return ori_imgs, unnormal_num, distence

def warnplay(ori_imgs, center_person, interval):
    # print(interval)
    tempAA = []
    if len(center_person) >= 2:
        for i in range(len(center_person)):
            ori_person = center_person[i, :]
            # other_person = np.delete(center_person, i, axis=0)
            temp = warn_or_normal(ori_person, center_person)
            mina = 1000
            for num in temp:
                if num > 0 and num < mina:
                    mina = num
            min_index = temp.index(mina)
            tempAA.append([i, min_index, temp[min_index]])
        # tempAA = np.array(tempAA)
        # tempAA = tempAA[np.lexsort(-tempAA.T)].tolist()
        tempAA_rm = tempAA.copy()
        for n in tempAA:
            arrol = [n[1], n[0], n[2]]
            if (arrol in tempAA_rm) and (n in tempAA_rm):
                tempAA_rm.remove(n)
        tempAA_rm = np.array(tempAA_rm)
        tempAA_rm = tempAA_rm[np.lexsort(-tempAA_rm.T)].tolist()
        tempAA = np.array(tempAA)
        tempAA = tempAA[np.lexsort(-tempAA.T)].tolist()
        img, unnormal_num, distence = drawimg(ori_imgs, tempAA_rm, center_person, interval)
    else:
        unnormal_num = 0
        distence = 0
        cv2.rectangle(ori_imgs, (10, 20), (550, 100), (0, 255, 255), thickness=-1)
        # cv2.putText(ori_imgs, 'Dist >  1m: ' + str(normal_num), (20, 80), cv2.FONT_HERSHEY_COMPLEX,  1.75, (255, 220, 0), 3)
        cv2.putText(ori_imgs, 'Dist <= ' + str(interval) + 'm: ' + str(unnormal_num), (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1.75, (100, 50, 255), 3)
        img = ori_imgs
    return img, unnormal_num, distence 
    
def change_lab(img, boxes, clss):
    end_lab = []
    if len(boxes) > 0:
        for i in range(len(boxes)):
            if clss[i] == 0:
                xx1, yy1, xx2, yy2 = boxes[i][0],\
                    boxes[i][1],\
                    boxes[i][2],\
                    boxes[i][3]
                x1 = int((xx2 + xx1) / 2)
                y1 = int(yy1)
                x2 = int((xx2 + xx1) / 2)
                y2 = int(yy2)
                end_lab.append([x1, y1, x2, y2])
    return end_lab

#####http####
def RobotLogin(deviceId, time, minDistance, logarithm, url, status):
    msg = collections.OrderedDict()
    msg = {'deviceId': deviceId, 'time': time, 'minDistance': minDistance, 'logarithm': logarithm, 'url': url, 'status': status}
    return msg

def cameraLogin(deviceId, status):
    msg = collections.OrderedDict()
    msg = {'deviceId': deviceId, 'status': status}
    return msg

def cameraurl(cameraid, url, arithmetic):
    msg = collections.OrderedDict()
    msg = {'deviceId': cameraid, 'url': url, 'arithmetic': arithmetic}
    return msg

def Post(url, msg):
    # print(msg)
    data_json = json.dumps(msg)
    data = "data=" + data_json
    data = requests.post(url=url, data=data, headers={'Content-Type':'application/x-www-form-urlencoded'})
    return data.text


class CarCamera(object):
    def __init__(self, cameraid, rount, alarmpost, deviceget):

        self.init_flag = True
        self.cameraid = cameraid
        self.rount = rount
        self.alarmpost = alarmpost
        self.deviceget = deviceget
        self.vs = cv2.VideoCapture(self.cameraid)
        self.grabbed, self.frame = self.vs.read()
        if not self.grabbed:
            Post(self.alarmpost, cameraLogin(self.rount, str(self.grabbed)))
        self.frame_queue = RQueue(max_len=5, name='Frame Queue.')
        self.camera_queue = RQueue(max_len=25, name='Camera Reading Buffer.')

    def get_frame(self):
        global put_img
        return cv2.imencode('.jpg', put_img)[1].tobytes()

    def camera_read(self):
        global camera_sta, dist_time, interval
        while True:
            try:
                grabbed, frame = self.vs.read()
                if grabbed:
                    camera_sta = True
                    self.camera_queue.put(frame)
                    try:
                        r = requests.get(self.deviceget)
                        r.encoding = 'utf-8'
                        dist_time = float(eval(r.text)['alarmInterval'])
                        interval = float(eval(r.text)['distance'])
                        print(interval)
                    except:
                        dist_time = 10
                        interval = 1
                else:
                    camera_sta = False
                    Post(self.alarmpost, cameraLogin(self.rount, str(camera_sta)))
                    self.vs = cv2.VideoCapture(self.cameraid)
            except Exception as e:
                print(traceback.format_exc())
                time.sleep(1)
            
    def feed_frames(self):
        """
        By Guanghao Zhang.
        直接将视频帧放入推送队列，不进行行人检测
        :return:
        """
        global unnormal_send, img_send, dist_time, interval, camera_sta, dist, put_img
        num = 0
        category_num = 80
        model = 'cfg/yolov3-416.trt'
        w,h = 416, 416
        conf_th = 0.7
        cuda.init()
        cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        trt_yolo = TrtYOLOv3(model, (h, w), category_num)
        while True:
            start = time.time()
            frame = self.camera_queue.get()
            boxes, confs, clss = trt_yolo.detect(frame, conf_th)
            end_lab = change_lab(frame, boxes, clss)
            end_img, unnormal_num, distence = warnplay(frame, np.array(end_lab), interval)
            end = time.time()
            print(end-start)
            put_img = end_img
            # self.frame_queue.put(end_img)
            if unnormal_num > unnormal_send:
                unnormal_send = unnormal_num 
                dist = distence
                img_send = end_img
            num += 1
            if int(num) % int(dist_time * 1000 / 100) == 0 and unnormal_send != 0 and camera_sta == True:
                if not os.path.exists('/DATA/pedestrian_dist/images/static/' + datetime.datetime.now().strftime("%Y-%m-%d")):
                    os.mkdir('/DATA/pedestrian_dist/images/static/' + datetime.datetime.now().strftime("%Y-%m-%d"))
                save_path = '/DATA/pedestrian_dist/images/static/' + datetime.datetime.now().strftime("%Y-%m-%d") + '/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '_' + str(self.rount) + '.jpg'
                # print(save_path)
                cv2.imwrite(save_path, img_send)
                Post(self.alarmpost, RobotLogin(self.rount, str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), str(round(dist, 1)), str(unnormal_send),
                        str(os.path.basename(save_path)), str(camera_sta)))
                num = 0
                unnormal_send = 0
        del trt_yolo
        cuda_ctx.pop()
        del cuda_ctx
            
    def run(self):  
        assert self.init_flag

        _cr_thread = threading.Thread(target=self.camera_read, )
        _cr_thread.daemon = True
        _cr_thread.start()
      
        _ff_thread = threading.Thread(target=self.feed_frames, )
        _ff_thread.daemon = True
        _ff_thread.start()
