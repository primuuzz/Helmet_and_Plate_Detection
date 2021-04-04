from flask import Flask, render_template, Response
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import re
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

app = Flask(__name__)

classes = ['Plat_Nomor',
          'Menggunakan_Helm',
          'Tidak_Menggunakan_Helm']

#facerecognition_model = "frozen_graph.pb"
net = cv2.dnn.readNetFromDarknet(r"E:\Prima\UNSOED\SKRIPSI_JUNI_WISUDA\pertemuan_9\4_Facerecognition_Mjpeg_Stream\yolo\helmet_detection_yolov3.cfg",r"E:\Prima\UNSOED\SKRIPSI_JUNI_WISUDA\pertemuan_9\4_Facerecognition_Mjpeg_Stream\yolo\yolov3_custom_final.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def draw_ped(img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):

    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img,
                  (x0, y0 + baseline),  
                  (max(xt, x0 + w), yt), 
                  color, 
                  2)
    cv2.rectangle(img,
                  (x0, y0 - h),  
                  (x0 + w, y0 + baseline), 
                  color, 
                  -1)  
    cv2.putText(img, 
                label, 
                (x0, y0),                   
                cv2.FONT_HERSHEY_SIMPLEX,     
                0.5,                          
                text_color,                
                1,
                cv2.LINE_AA) 
    return img

def recognize_face(frame):
    img = cv2.resize(frame,(900,700))
    hight,width,_ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = False,crop= False)

    net.setInput(blob)
    
    layerOutput = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(layerOutput)

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


    print(len(confidences))
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.6,.4)

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

    
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.6,.4)
    print(indexes)
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

    #cv2.imshow('img',img)
    return img

def gen_frames():  
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = recognize_face(frame)
                     
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')


camera = cv2.VideoCapture(0)
app.run()