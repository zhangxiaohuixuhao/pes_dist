# -*- coding: utf-8 -*-  

from flask import Flask, Response, jsonify, render_template, request

import argparse
from camera_v3rt import CarCamera
import os
import datetime
from xml.dom import minidom
from camera_v3rt import *
import socket

app = Flask(__name__)

dom = minidom.parse('./car.xml')

for ele in dom.getElementsByTagName('Camera'):
    cameraid = ele.getElementsByTagName('id')[0].firstChild.data
    flask_ip = ele.getElementsByTagName('localip')[0].firstChild.data
    flask_port = int(ele.getElementsByTagName('flaskport')[0].firstChild.data)
    rount = ele.getElementsByTagName('rount')[0].firstChild.data
    alarmpost = ele.getElementsByTagName('alarmpost')[0].firstChild.data
    devicepost = ele.getElementsByTagName('devicepost')[0].firstChild.data
    deviceget = ele.getElementsByTagName('deviceget')[0].firstChild.data
print(cameraid, flask_ip, flask_port, rount, alarmpost, devicepost, deviceget)
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.connect(('8.8.8.8',80))
ip=s.getsockname()[0]
print(ip)
s.close()
Post(devicepost, cameraurl(rount, 'http://' + ip + ':' + str(flask_port) + '/' + rount, 'True'))

car_obj = CarCamera(cameraid=cameraid, rount=rount, alarmpost=alarmpost, deviceget=deviceget)

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        # print(datetime.datetime.now())
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/' + rount)
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(car_obj),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get/<dist>')# 访问http://127.0.0.1:8090/get/563249423 就会返回我提交的数据
def get(dist):
    return "<h1>提交的数据为：{0}</h1>".format(dist)

if __name__ == '__main__':
    car_obj.run()
    app.debug = False
    app.run(host='0.0.0.0', port=int(flask_port), threaded=True)