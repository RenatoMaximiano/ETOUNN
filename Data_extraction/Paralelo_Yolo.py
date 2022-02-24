from threading import Thread
import time
import pip

#Video Yolo todos
#Importando bibliotecas
import numpy as np
import cv2
import time
# Variables to define
CFG = '.cfg'
WEIGHTS = '.weights'
NAMES = '.names'


Confidence = 0.4

VIDEO='input' #Use camera = 0

IMSHOW = 0 #Yes = 1 No = 0
FPSSHOW = 1 #Yes = 1 No = 0
IMSAVE = 1 # Yes = 1 No = 0
OUTPUT = "output"

ISO_CLASS = 1
INDEX = 2

BLOB = (320,320)# 416, 512
#######################################DO NOT CHANGE##################################################################
def func1():
                def extract_boxes_confidences_classids(outputs, confidence, width, height):
                    boxes = []
                    confidences = []
                    classIDs = []

                    for output in outputs:
                        for detection in output:            
                            # Extract the scores, classid, and the confidence of the prediction
                            scores = detection[5:]
                            classID = np.argmax(scores)
                            conf = scores[classID]
                            
                            # Consider only the predictions that are above the confidence threshold
                            if conf > confidence:
                                # Scale the bounding box back to the size of the image
                                box = detection[0:4] * np.array([width, height, width, height])
                                centerX, centerY, w, h = box.astype('int')

                                # Use the center coordinates, width and height to get the coordinates of the top left corner
                                x = int(centerX - (w / 2))
                                y = int(centerY - (h / 2))

                                boxes.append([x, y, int(w), int(h)])
                                confidences.append(float(conf))
                                classIDs.append(classID)

                    return boxes, confidences, classIDs


                def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors):
                    if len(idxs) > 0:
                        for i in idxs.flatten():
                            if confidences[i] > Confidence:
                             if ISO_CLASS == 1:
                               if classIDs[i] == INDEX:
                                x, y = boxes[i][0], boxes[i][1]
                                w, h = boxes[i][2], boxes[i][3]

                            # draw the bounding box and label on the image
                                color = [int(c) for c in colors[classIDs[i]]]
                                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                             else:                  
                            # extract bounding box coordinates
                              x, y = boxes[i][0], boxes[i][1]
                              w, h = boxes[i][2], boxes[i][3]

                            # draw the bounding box and label on the image
                              color = [int(c) for c in colors[classIDs[i]]]
                              cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                              text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                              cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            if FPSSHOW == 1:
                             cv2.putText(image, "FPS: "+str(round(Fps,2)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)

                    return image

                def make_prediction(net, layer_names, labels, image, confidence, threshold):
                    height, width = image.shape[:2]
                    # Pre-process the image to make it blob
                    # Go through the Yolo model
                    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, BLOB, swapRB=True, crop=False)
                    net.setInput(blob)
                    outputs = net.forward(layer_names)

                    # Extract the rectangles, trust and classIDs
                    boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, confidence, width, height)

                    # Apply Non-Max Suppression
                    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

                    return boxes, confidences, classIDs, idxs

                # Objects that the model detects
                labels = open(NAMES).read().strip().split('\n')

                #Randomly generate colors for each object category
                colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

                # Load the model and weights
                net = cv2.dnn.readNetFromDarknet(CFG, WEIGHTS)

                use_gpu = 0
                if (use_gpu == 1):
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

                # Get the name of the categories

                layer_names = net.getLayerNames()

                layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                print(layer_names)
                cap = cv2.VideoCapture(VIDEO)
                #cap = cv2.VideoCapture(0)
                stop = 0
                Fps=0
                i=0
                while (True):
                    s=time.time()
                    i=i+1
                    if(stop == 0):
                        ret, frame = cap.read()
                        if (ret == True):
                            boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, frame, 0.1, 0.3)
                            frame = draw_bounding_boxes(frame, boxes, confidences, classIDs, idxs, colors)
                            if IMSHOW == 1:
                             cv2.imshow("Frame", frame)
                            Fps=(1/(time.time()-s))
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
                workbook = xlsxwriter.Workbook('Dados/Yolov3-416/Dados3/Dados.xlsx')
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