from threading import Thread
import time
import pip
import numpy as np
import cv2
import time
# Variaveis para definir
Confidence = 0.4
VIDEO='videocar.mp4' #Use camera = 0

IMSHOW = 0 #Yes = 1 No = 0
FPSSHOW = 1 #Yes = 1 No = 0
IMSAVE = 1 # Yes = 1 No = 0
OUTPUT = "Datas/mask-r-cnn/"

ISO_CLASS = 1
INDEX = 0
def func1():
                def prediction(frame, net, CLASSES, COLORS, confidence_lim):
                    (frameH, frameW) = frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
                    #blob = cv2.dnn.blobFromImage(frame,1/255, size=(32, 32), swapRB=True, crop=False)
                    net.setInput(blob)
                    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
                    numClasses = masks.shape[1]
                    numDetections = boxes.shape[2]
                    boxesToDraw = []
                    for i in range(numDetections):
                        box = boxes[0, 0, i]
                        mask = masks[i]
                        score = box[2]
                        if score > Confidence:
                            classId = int(box[1])
                            if ISO_CLASS == 1:
                                if classId == INDEX:
                                    left = int(frameW * box[3])
                                    top = int(frameH * box[4])
                                    right = int(frameW * box[5])
                                    bottom = int(frameH * box[6])
                                    left = max(0, min(left, frameW - 1))
                                    top = max(0, min(top, frameH - 1))
                                    right = max(0, min(right, frameW - 1))
                                    bottom = max(0, min(bottom, frameH - 1))
                                    boxesToDraw.append([frame, classId, score, left, top, right, bottom])
                                    classMask = mask[classId]
                                    classMask = cv2.resize(classMask, (right - left + 1, bottom - top + 1))
                                    mask = (classMask > 0.4)
                                    roi = frame[top:bottom+1, left:right+1][mask]
                                    frame[top:bottom+1, left:right+1][mask] = (0.7 * COLORS[classId] + 0.3 * roi).astype(np.uint8)
                                    cv2.rectangle(frame, (left, top), (right, bottom), COLORS[classId], 3)
                                    label = "{}: {:.2f}%".format(CLASSES[classId], score * 100)
                                    y = top - 15 if top - 15 > 15 else top + 15
                                    cv2.putText(frame, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[classId], 2)
                            else:
                                            left = int(frameW * box[3])
                                            top = int(frameH * box[4])
                                            right = int(frameW * box[5])
                                            bottom = int(frameH * box[6])

                                            left = max(0, min(left, frameW - 1))
                                            top = max(0, min(top, frameH - 1))
                                            right = max(0, min(right, frameW - 1))
                                            bottom = max(0, min(bottom, frameH - 1))

                                            boxesToDraw.append([frame, classId, score, left, top, right, bottom])

                                            classMask = mask[classId]
                                            classMask = cv2.resize(classMask, (right - left + 1, bottom - top + 1))
                                            mask = (classMask > 0.4)

                                            roi = frame[top:bottom+1, left:right+1][mask]
                                            frame[top:bottom+1, left:right+1][mask] = (0.7 * COLORS[classId] + 0.3 * roi).astype(np.uint8)

                                            cv2.rectangle(frame, (left, top), (right, bottom), COLORS[classId], 3)
                                            label = "{}: {:.2f}%".format(CLASSES[classId], score * 100)
                                            y = top - 15 if top - 15 > 15 else top + 15
                                            cv2.putText(frame, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[classId], 2)
                                


                    return frame
                #CLASSES = ["person"]
                CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush" ]
                COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

                net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')
                #net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph2.pb', 'faster_rcnn_inception_v2_coco_2018_01_28.pbtxt')

                use_gpu = 0
                if (use_gpu == 1):
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                confidence_lim = 0.4

                cap = cv2.VideoCapture(VIDEO)
                #cap = cv2.VideoCapture(0)
                i=0

                stop = 0
                while (True):
                    s=time.time()
                    i=i+1
                    if(stop == 0):
                        ret, frame = cap.read()
                        if (ret == True):
                            frame = prediction(frame, net, CLASSES, COLORS, confidence_lim)
                            if IMSHOW == 1:
                                cv2.imshow("Frame", frame)
                            #print(CLASSES)
                            #print(1/(time.time()-s))
                            #if i <= 100 or 500 >= i >= 400:
                            if IMSAVE == 1:
                                cv2.imwrite(OUTPUT+str(i)+".png",frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        stop = not(stop)
                    if key == ord('q'):
                        break
                cv2.destroyAllWindows()
                    
   

def func2():
                import os
                import time
                def getCPUtemperature():
                    res = os.popen('vcgencmd measure_temp').readline()
                    return(res.replace("temp=","").replace("'C\n",""))

                CPU_temp = float(getCPUtemperature())


                def getRAMinfo():
                    p = os.popen('free')
                    i = 0
                    while 1:
                        i = i + 1
                        line = p.readline()
                        if i==2:
                            return(line.split()[1:4])
                RAM_stats = getRAMinfo()
                RAM_used = round(int(RAM_stats[1]) / 1000,1)
                #print(RAM_used )

                import psutil
                def measure_temp():
                        temp = os.popen("vcgencmd measure_temp").readline()
                        return (temp.replace("temp=",""))
                # you can have the percentage of used RAM
                #expenses=[["TIME","TEMP","CPU","MEM"]]
                expenses=[["TEMP","CPU"]]
                #j=0
                for i in range(1800):
                    #s=time.time()
                    #j=j+1
                    cpu=(psutil.cpu_percent(1))
                    #RAM_stats = getRAMinfo()
                    #RAM_used = round(int(RAM_stats[1]) / 1000,1)
                    CPU_temp = float(getCPUtemperature())
                    #Time= time.strftime("%H:%M:%S")
                    #a=[Time,CPU_temp,cpu,RAM_used]
                    a=[CPU_temp,cpu]
                    expenses.append(a)
                    #print(1/(time.time()-s))
                    print(a)
                    print(i)
                    #print(expenses)
                  


                


                import xlsxwriter
                import pandas as pd
                print(1)


                # Create a workbook and add a worksheet.
                workbook = xlsxwriter.Workbook('Dados/mask-r-cnn/Dados1/Dados.xlsx')
                worksheet = workbook.add_worksheet()

                # Some data we want to write to the worksheet.


                # Start from the first cell. Rows and columns are zero indexed.
                row = 0
                col = 0

                # Iterate over the data and write it out row by row.
                for item, cost in (expenses):
                    worksheet.write(row, col,     item)
                    worksheet.write(row, col + 1, cost)
                    #worksheet.write(row, col + 2, sla)
                    #worksheet.write(row, col + 3, sla2)
                    row += 1


                workbook.close()

if __name__ == '__main__':
    Thread(target = func1).start()
    Thread(target = func2).start()