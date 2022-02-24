# ETOUNN
<p align="center">
  <img src="https://user-images.githubusercontent.com/84810481/155543417-51c626b9-6250-493c-9a97-64516ab42df7.png">
</p>
<p align="center">
The proposed method, called Analyzing Energy Consumption and Temperature of on-board computer of UAVs via Neural Networks (ETOUNN), can be divided in 6 steps.
</p>
<p align="center">
 <img src="https://user-images.githubusercontent.com/84810481/155544944-cec2f9a3-772d-4ae7-8842-86f10175ccae.png">
</p>

# Initial tests and configurations
The first step is to configure the Raspberry PI to run the object detection algorithms. In our studies we used a Raspberry Pi 4 8Gb, in this case we recommend version 4.5.0 of OpenCv that can be installed through: [OpenCV-4-5-0](https://github.com/RenatoMaximiano/ETOUNN/blob/main/OpenCV-4-5-0.sh).
We use known, pre-trained algorithms from the YOLO family developed [1](https://github.com/AlexeyAB/darknet) and Mask R-CNN developed by [2](https://github.com/escoladeestudantes/opencv/tree/main/22_ObjectDetection_Mask-RCNN_Inception_v2_COCO). The pre-trained models that we use are in the following folder: [pre-trained](https://drive.google.com/drive/folders/1fAw1LLeFAWj5KCKIHldxqFhxjPC5ZnkI?usp=sharing)

# Data extraction for energy estimation model
For data extraction, the following algorithms were developed [Data extraction - Rasp/YOLO](https://github.com/RenatoMaximiano/ETOUNN/blob/main/Data_extraction/Paralelo_Yolo.py) for YOLO and [Data extraction - Rasp/Mask](https://github.com/RenatoMaximiano/ETOUNN/blob/main/Data_extraction/Paralelo_Mask.py) for Mask R-CNN. These algorithms perform the task of extracting data from the Raspberry Pi at the same time as detecting objects.

The data extracted and used for training the models and analysis can be obtained here: [DATA](https://github.com/RenatoMaximiano/ETOUNN/tree/main/DATA)
