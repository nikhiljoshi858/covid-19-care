from django.shortcuts import render
from django.core.files import File
from django.http import HttpResponse, JsonResponse
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
from base64 import b64encode, decodestring, decodebytes, b64decode
from datetime import datetime
from account.models import Previous_Mask
import pytz
from ipstack import GeoLookup
from .models import Video
import csv
import xlwt
from django.views.decorators.csrf import csrf_exempt
# import vlc
# Create your views here.

prototxtPath = 'D:\Django_Projects/temp/models/deploy.prototxt'
weightsPath = 'D:\Django_Projects/temp/models/resnet.caffemodel'


global location
geo_lookup = GeoLookup('776da34f4f37c2fb8f3ad306cc615bff')
location = geo_lookup.get_own_location()
location = location['city'] + ', ' + location['region_name'] + ', ' + location['country_name']


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

                headers = {'content-type': 'applications/json'}
                data = json.dumps({'signature_type': 'serving_default', 'instances': face.tolist()})
                json_response = requests.post('http://localhost:8501/v1/models/temp:predict', data=data, headers=headers)
                predictions = json.loads(json_response.text)
                mask, without_mask = predictions['predictions'][0]

                # model = load_model('D:/Django_Projects/temp/models/temp')
                # print('[INFO] Loading Saved model...')
                # mask, without_mask = model.predict(face)[0]

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

        uri = b64encode(cv2.imencode('.jpg', cv2.resize(image, (300,300)))[1]).decode()
        uri = "data:%s;base64,%s" % ("image/jpeg", uri)

        if label_image == 'Mask':
            row = Previous_Mask(timestamp=datetime.now(pytz.timezone('Asia/Kolkata')), result='Wearing Mask', category='image', location=location)
        else:
            row = Previous_Mask(timestamp=datetime.now(pytz.timezone('Asia/Kolkata')), result='Not Wearing Mask', category='image', location=location)
        row.save()

        context = {}
        context['image'] = uri
        context['label'] = label

        return render(request, 'mask/mask_image_output.html', context)
    return render(request, 'mask/image.html')



@login_required(redirect_field_name='mask/video/')
def video_view(request):
    if request.method == 'POST':
        # print(request.POST)
        # print(request.FILES['video'])
        video = request.FILES['video']
        v = Video(video=video)
        v.save()

        v = Video.objects.last()
        # return HttpResponse(v.video.url)
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        # return HttpResponse(v.video.url)
        video = cv2.VideoCapture(settings.BASE_DIR+str(v.video.url))
        # if video.isOpened() == False:
        #     width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        #     height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # size = (int(width), int(height))
        # return HttpResponse(size)
        counter = 0
        maskNet = load_model('D:\Django_Projects/temp/models/temp')
        while True:
            grabbed, frame = video.read()
            if not grabbed:
                break
            # cv2.imshow('frame', frame)
            if counter == 0:
                width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
                video_file = cv2.VideoWriter('mask_output.avi', cv2.VideoWriter_fourcc(*"MJPG"), 25, (frame.shape[1], frame.shape[0]))
                counter = 1
            frame = imutils.resize(frame, width=800)
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                if mask > withoutMask:
                    label_webcam = "Mask"
                    color = (0, 255, 0)
                else:
                    label_webcam =  "No Mask"
                    color = (0, 0, 255)
                label= "{}: {:.2f}%".format(label_webcam, max(mask, withoutMask) * 100)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            video_file.write(frame)
        
        # v = Video(video=video_file)
        # v.save()
        context = {}
        context['video'] = video_file
        return render(request, 'mask/video_output.html', context=context)
    return render(request, 'mask/video.html')





@login_required(redirect_field_name='mask/webcam/')
@csrf_exempt
def webcam_view(request):
    # prototxtPath = settings.BASE_DIR+'/models/deploy.prototxt'
    # weightsPath = settings.BASE_DIR+'/models/res10_300x300_ssd_iter_140000.caffemodel'
    if request.method == 'POST':
        uri = request.POST['uri']
        img_data = b64decode(uri.split(',')[1])

        nparr = np.fromstring(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        prototxtPath = settings.BASE_DIR+'/models/deploy.prototxt'
        weightsPath = settings.BASE_DIR+'/models/res10_300x300_ssd_iter_140000.caffemodel'
        # prototxtPath = 'E:\Django_Projects/temp/models/deploy.prototxt'
        # weightsPath = 'E:\Django_Projects/temp/models/resnet.caffemodel'

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

                headers = {'content-type': 'applications/json'}
                data = json.dumps({'signature_type': 'serving_default', 'instances': face.tolist()})
                json_response = requests.post('http://localhost:8501/v1/models/mask:predict', data=data, headers=headers)
                predictions = json.loads(json_response.text)
                mask, without_mask = predictions['predictions'][0]

                # model = load_model('D:/Django_Projects/temp/models/temp')
                # print('[INFO] Loading Saved model...')
                # mask, without_mask = model.predict(face)[0]

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
        mask = False
        if label_image == 'Mask':
            mask = True
            row = Previous_Mask(timestamp=datetime.now(pytz.timezone('Asia/Kolkata')), result='Wearing Mask', category='image', location=location)
        else:
            row = Previous_Mask(timestamp=datetime.now(pytz.timezone('Asia/Kolkata')), result='Not Wearing Mask', category='image', location=location)
            mask = False
        row.save()

        context = {}
        context['image'] = uri
        context['label'] = label
        return JsonResponse({'image':uri, 'mask':mask})
    
    return render(request, 'mask/webcam.html')


def previous_results_image_csv_view(request):
    rows = Previous_Mask.objects.filter(category='image')
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="Previous results on mask detection in images.csv"'
    writer = csv.writer(response)
    writer.writerow(['Sr no.', 'Location', 'Date and Time', 'Result'])
    for row in rows:
        writer.writerow([row.id, row.location, row.timestamp, row.result])
    return response

def previous_results_view(request):
    context = {}
    objects = Previous_Mask.objects.all().order_by('timestamp')
    context['previous'] = objects
    return render(request, 'mask/previous.html', context=context)

def previous_results_video_csv_view(request):
    rows = Previous_Mask.objects.filter(category='video')
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="Previous results on mask detection in videos.csv"'
    writer = csv.writer(response)
    writer.writerow(['Sr no.', 'Location', 'Date and Time', 'Result'])
    for row in rows:
        writer.writerow([row.id, row.location, row.timestamp, row.result])
    return response


def previous_results_image_xlsx_view(request):
    rows = Previous_Mask.objects.filter(category='image')
    response = HttpResponse(content_type='application/ms-excel')
    response['Content-Disposition'] = 'attachment; filename="Previous results on mask detection in images.xls"'
    wb = xlwt.Workbook(encoding='utf-8')
    ws = wb.add_sheet('Sheet 1')
    row_num = 0
    font_style = xlwt.XFStyle()
    font_style.font.bold = True
    columns = ['Sr no.', 'Location', 'Date and Time', 'Result']
    for col_num in range(len(columns)):
        ws.write(row_num, col_num, columns[col_num], font_style)
    font_style = xlwt.XFStyle()
    for row in rows:
        row_num = row_num + 1
        ws.write(row_num, 0, row.id, font_style)
        ws.write(row_num, 1, row.location, font_style)
        ws.write(row_num, 2, str(row.timestamp), font_style)
        ws.write(row_num, 3, row.result, font_style)

    wb.save(response)
    return response
    

def previous_results_video_xlsx_view(request):
    rows = Previous_Mask.objects.filter(category='video')
    response = HttpResponse(content_type='application/ms-excel')
    response['Content-Disposition'] = 'attachment; filename="Previous results on mask detection in videos.xls"'
    wb = xlwt.Workbook(encoding='utf-8')
    ws = wb.add_sheet('Sheet 1')
    row_num = 0
    font_style = xlwt.XFStyle()
    font_style.font.bold = True
    columns = ['Sr no.', 'Location', 'Date and Time', 'Result']
    for col_num in range(len(columns)):
        ws.write(row_num, col_num, columns[col_num], font_style)
    font_style = xlwt.XFStyle()
    for row in rows:
        row_num = row_num + 1
        ws.write(row_num, 0, row.id, font_style)
        ws.write(row_num, 1, row.location, font_style)
        ws.write(row_num, 2, str(row.timestamp), font_style)
        ws.write(row_num, 3, row.result, font_style)

    wb.save(response)
    return response
    