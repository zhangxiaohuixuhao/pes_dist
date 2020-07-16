# 行人社交距离检测算法

## 环境依赖

```
conda env create -f condaenvir.yaml
```

```
pip install -r requirements.txt
```

```
TensorRT-6.0.1.5
```

## 代码运行说明

#### 主程序：

```
python app.py
```

#### 其中：

```
camera_v3.py 使用yolov3模型，调用darknet_v3.py，模型等文件在cfg_v3中
```

```
camera_v4.py 使用yolov4模型，调用darknet_v4.py，修改模型可以调用yolov4-tiny模型，模型等文件在cfg中
```

```
camera_v3rt.py 使用yolov3模型tensorrt转换模型，模型文件在cfg中
```

