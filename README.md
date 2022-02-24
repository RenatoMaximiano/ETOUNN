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

#  Energy consumption estimation model
The objective of the first model is to estimate the energy consumption of raspberry PI. Since the voltage is constant in 5.15 V, we needed to estimate the electrical current. Through CPU and temperature data using a multilayer perceptron neural network (MLP). the Model can be seen in the following image and the algorithm for the training can be obtained here: [MLP_MODEL](https://github.com/RenatoMaximiano/ETOUNN/blob/main/Models/mlp_finalipynb.py)

<p align="center">
  <img src="https://user-images.githubusercontent.com/84810481/155583888-23c20e22-c044-4c58-a808-f751ed5b5d40.png">
</p>

# Temperature prediction model
The second model aims to predict the temperature of the Raspberry Pi processor. For this, a Long Short-Term Memory (LSTM) network was used. The model topology can be seen below and the training algorithm here:

<p align="center">
  <img src="https://user-images.githubusercontent.com/84810481/155585545-2d894252-54e5-48a7-80eb-adf8f76df02b.png">
</p>
