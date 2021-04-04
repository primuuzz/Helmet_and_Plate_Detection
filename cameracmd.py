import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import re
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

net = cv2.dnn.readNetFromDarknet("helmet_detection_yolov3.cfg",r"C:\Users\user\Downloads\yolov3_costum_final.weights")
classes = ['Plat Nomor','Menggunakan Helm','Tidak Menggunakan Helm',]

def ocr(img, box):
    x, y, w, h = box
    crop = img[y:y+h, x:x+w, :]
    crop = cv2.resize(crop, None, fx=2, fy=2)
    #cv2_imshow(crop)
    # Alternatively: can be skipped if you have a Blackwhite image
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray = cv2.bitwise_not(img_bin)
    # Removing noise by morphological operations:
    kernel = np.ones((3, 3), np.uint8)
    crop = cv2.erode(gray, kernel, iterations=1)
    crop = cv2.dilate(crop, kernel, iterations=1)
    #cv2.imshow('test',crop)
    # Recognizing characters:
    return pytesseract.image_to_string(crop)
    #print (output)
    # Optimizing output:
    #lines = re.split("\n+", output)
    #new_lines = []
    #for line in lines:
    #    words = re.split("\\s+", line.strip().upper())
    #    new_line = " ".join(words)
    #if new_line not in ("", "W", "WO", "OW"):
    #    new_lines.append(new_line)
    #    new_output = "\n".join(new_lines)
    #    return new_output
    
cap = cv2.VideoCapture(0)

while 1:
    _, img = cap.read()
    img = cv2.resize(img,(900,700))
    hight,width,_ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

    net.setInput(blob)

    output_layers_name = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(output_layers_name)

    boxes =[]
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3]* hight)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.5,.4)

    boxes =[]
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3]* hight)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.8,.4)
    # Non-maximum suppression:
    results = [(class_ids[i], boxes[i]) for i in range(len(boxes)) if i in indexes]
    
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size =(len(boxes),3))
    if  len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            if (label == 'Plat Nomor'):
                cv2.putText(img,ocr(img,boxes[i]) + "  " + confidence, (x,y+200),font,2,color,2)
                print(ocr(img,boxes[i]))
            else:
                cv2.putText(img,label + " " + confidence, (x,y+400),font,2,color,2)

    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()