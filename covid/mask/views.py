from django.shortcuts import render
from django.core.files import File
from django.http import HttpResponse
from django.conf import settings
from django.contrib.auth.decorators import login_required
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import json
import requests
from imutils.video import VideoStream
import imutils
import time
from base64 import b64encode
from datetime import datetime
from account.models import Previous_Mask
import pytz
# import vlc
# Create your views here.



def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)



@login_required(redirect_field_name='mask/')
def homepage_view(request):
    return render(request, 'mask/home.html')


@login_required(redirect_field_name='mask/image/')
def image_view(request):
    if request.method == "POST":
        image = cv2.imdecode(np.fromstring(request.FILES['files'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # prototxtPath = settings.BASE_DIR+'/models/deploy.prototxt'
        # weightsPath = settings.BASE_DIR+'/models/res10_300x300_ssd_iter_140000.caffemodel'
        prototxtPath = 'D:/Django_Projects/temp/models/deploy.prototxt'
        weightsPath = 'D:/Django_Projects/temp/models/resnet.caffemodel'


        net = cv2.dnn.readNet(prototxtPath, weightsPath)

        orig = image.copy()
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence > 0.5:
                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                startx, starty, endx, endy = box.astype('int')
                startx, starty = max(0,startx), max(0,starty)
                endx, endy = min(w-1, endx), min(h-1, endy)

                face = image[starty:endy, startx:endx]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224,224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                # headers = {'content-type': 'applications/json'}
                # data = json.dumps({'signature_type': 'serving_default', 'instances': face.tolist()})
                # json_response = requests.post('http://localhost:8501/v1/models/facemask:predict', data=data, headers=headers)
                # predictions = json.loads(json_response.text)
                # mask, without_mask = predictions['predictions'][0]

                model = load_model('D:/Django_Projects/temp/models/temp')
                print('[INFO] Loading Saved model...')
                mask, without_mask = model.predict(face)[0]

                global label_image

                if mask > without_mask:
                    label_image = 'Mask'
                    color = (0, 255, 0)
                else:
                    label_image = 'No Mask'
                    color = (0, 0, 255)

                label = '{}: {:.2f}%'.format(label_image, max(mask, without_mask) * 100)

                cv2.putText(image, label, (startx, starty-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (startx, starty), (endx, endy), color, 2)

        uri = b64encode(cv2.imencode('.jpg', image)[1]).decode()
        uri = "data:%s;base64,%s" % ("image/jpeg", uri)

        if label_image == 'Mask':
            row = Previous_Mask(timestamp=datetime.now(pytz.timezone('Asia/Kolkata')), result='Wearing Mask', category='image')
        else:
            row = Previous_Mask(timestamp=datetime.now(pytz.timezone('Asia/Kolkata')), result='Not Wearing Mask', category='image')
        row.save()

        context = {}
        context['image'] = uri
        context['label'] = label

        return render(request, 'mask/mask_image_output.html', context)
    return render(request, 'mask/image.html')



@login_required(redirect_field_name='mask/video/')
def video_view(request):

    if request.method == 'POST':
        video = request.FILES['files'].read()
        cap = cv2.VideoCapture(str(video))
        if (cap.isOpened()== False):  
            return HttpResponse("Error opening video file")
        while(cap.isOpened()): 
            ret, frame = cap.read() 
            if ret == True: 
                cv2.imshow('Frame', frame) 
                if cv2.waitKey(25) & 0xFF == ord('q'): 
                    break    
            else:  
                break
        # return HttpResponse(str(request.FILES['files'].read()))
    return render(request, 'mask/video.html')



def previous_results_view(request):
    context = {}
    objects = Previous_Mask.objects.all().order_by('-timestamp')
    context['previous'] = objects
    return render(request, 'mask/previous.html', context=context)


@login_required(redirect_field_name='mask/webcam/')
def webcam_view(request):
    prototxtPath = settings.BASE_DIR+'/models/deploy.prototxt'
    weightsPath = settings.BASE_DIR+'/models/res10_300x300_ssd_iter_140000.caffemodel'


    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    print("[INFO] loading face mask detector model...")
    maskNet = load_model(settings.BASE_DIR+'/models')
    print("[INFO] starting video stream...")

    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=800)

        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            global label_webcam

            # player = vlc.MediaPlayer('/home/nikhil/Desktop/Django_Projects/covid/covid/mask/static/alarm.mp3')
            if mask > withoutMask:
                label_webcam = "Mask"
                color = (0, 255, 0)
                # player.play()
            else:
                label_webcam =  "No Mask"
                color = (0, 0, 255)
                # player.pause()

            label= "{}: {:.2f}%".format(label_webcam, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


    cv2.destroyAllWindows()
    vs.stop()
    
    if label_webcam == 'Mask':
        row = Previous_Mask(timestamp=datetime.now(pytz.timezone('Asia/Kolkata')), result='Wearing Mask', category='video')
    else:
        row = Previous_Mask(timestamp=datetime.now(pytz.timezone('Asia/Kolkata')), result='Not Wearing Mask', category='video')
    row.save()

    return render(request, 'mask/home.html')
