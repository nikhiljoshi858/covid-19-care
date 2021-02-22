from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.core.files import File
import numpy as np
import cv2
from scipy.spatial import distance as dist
import imutils
import os
from base64 import b64encode
from datetime import datetime
from account.models import Previous_Social
import pytz
from ipstack import GeoLookup
import csv
import xlwt
from .models import Video


# Create your views here.


FOCAL_LENGTH = 990
MIN_CONF = 0.3
NMS_THRESH = 0.3
MIN_DISTANCE = 180
PPI = 140
WIDTH = 41.1


global location
geo_lookup = GeoLookup('776da34f4f37c2fb8f3ad306cc615bff')
location = geo_lookup.get_own_location()
location = location['city'] + ', ' + location['region_name'] + ', ' + location['country_name']


def detect_people(frame, net, ln, personIdx=0):
	(H, W) = frame.shape[:2]
	results = []

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	boxes = []
	centroids = []
	confidences = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personIdx and confidence > MIN_CONF:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			r = (confidences[i], (x, y, x + w, y + h, 41.1*2.54*FOCAL_LENGTH/w), centroids[i])
			results.append(r)

	return results


def previous_results_view(request):
    context = {}
    objects = Previous_Social.objects.all().order_by('timestamp')
    context['previous'] = objects
    return render(request, 'social/previous.html', context=context)


@login_required(redirect_field_name='/social/')
def homepage_view(request):
    return render(request, 'social/home.html')


@login_required(redirect_field_name='/social/image')
def image_view(request):
    if request.method == "POST":
        img = cv2.imdecode(np.fromstring(request.FILES['files'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        labelsPath = 'D:\Django_Projects/temp/models/coco.names'
        LABELS = open(labelsPath).read().strip().split("\n")

        weightsPath = 'D:\Django_Projects/temp/models/yolov3.weights'
        configPath = 'D:\Django_Projects/temp/models/yolov3.cfg'

        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        results = detect_people(img, net, ln, personIdx=LABELS.index("person"))

        violate = set()
        print(results)

        if len(results) >= 2:
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean").tolist()

            for i in range(len(D[0])):
                for j in range(i + 1, len(D[1])):
                    print('Stage 1:',D[i][j])
                    D[i][j] = (D[i][j]  / PPI) * 2.54
                    print('Stage 2:',D[i][j])
                    D[i][j] = (D[i][j]**2 + (results[i][1][4] - results[j][1][4])**2) ** 0.5
                    print('Stage 3:',D[i][j])
                    print()
                    if D[i][j] < MIN_DISTANCE:
                        violate.add(i)
                        violate.add(j)

        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY, di) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            if i in violate:
                color = (0, 0, 255)

            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
            cv2.circle(img, (cX, cY), 5, color, 1)

        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.85, (0, 0, 255), 3)

        uri = b64encode(cv2.imencode('.jpg', img)[1]).decode()
        uri = "data:%s;base64,%s" % ("image/jpeg", uri)

        row = Previous_Social(result=len(violate), category='image', location=location)
        row.save()

        context = {}
        context['image'] = uri
        context['number'] = len(violate)

        return render(request, 'social/image_output.html', context)


    return render(request, 'social/image.html')


@login_required(redirect_field_name='/social/video')
def video_view(request):
    if request.method == 'POST':
        video = request.FILES['video']
        # video = request.POST['video']
        v = Video(video=video)
        v.save()

        v = Video.objects.last()
        context = {}
        context['video'] = v
        return render(request, 'social/video_output.html', context=context)

    return render(request, 'social/video.html')


@login_required(redirect_field_name='/social/webcam')
def webcam_view(request):
    labelsPath = 'E:\Django_Projects/temp/models/coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")

    weightsPath = 'E:\Django_Projects/temp/models/yolov3.weights'
    configPath = '/home/nikhil/Desktop/models/yolov3.cfg'

    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    print("[INFO] accessing video stream...")
    vs = cv2.VideoCapture(0)

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        frame = imutils.resize(frame, width=1000)
        results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

        violate = set()

        if len(results) >= 2:
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < MIN_DISTANCE:
                        violate.add(i)
                        violate.add(j)

        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            if i in violate:
                color = (0, 0, 255)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_COMPLEX, 0.85, (0, 0, 255), 3)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


    row = Previous_Social(result=len(violate), category='video', location=location)
    row.save()


    return render(request, 'social/webcam.html')



def previous_results_image_csv_view(request):
    rows = Previous_Social.objects.filter(category='image')
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="Previous results on Social Distancing detection in images.csv"'
    writer = csv.writer(response)
    writer.writerow(['Sr no.', 'Location', 'Date and Time', 'No. of Violations'])
    for row in rows:
        writer.writerow([row.id, row.location, row.timestamp, row.result])
    return response


def previous_results_video_csv_view(request):
    rows = Previous_Social.objects.filter(category='video')
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="Previous results on Social Distancing detection in videos.csv"'
    writer = csv.writer(response)
    writer.writerow(['Sr no.', 'Location', 'Date and Time', 'No. of Violations'])
    for row in rows:
        writer.writerow([row.id, row.location, row.timestamp, row.result])
    return response
    

def previous_results_image_xlsx_view(request):
    rows = Previous_Social.objects.filter(category='image')
    response = HttpResponse(content_type='application/ms-excel')
    response['Content-Disposition'] = 'attachment; filename="Previous results on Social Distancing detection in images.xls"'
    wb = xlwt.Workbook(encoding='utf-8')
    ws = wb.add_sheet('Sheet 1')
    row_num = 0
    font_style = xlwt.XFStyle()
    font_style.font.bold = True
    columns = ['Sr no.', 'Location', 'Date and Time', 'No. of Violations']
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
    rows = Previous_Social.objects.filter(category='video')
    response = HttpResponse(content_type='application/ms-excel')
    response['Content-Disposition'] = 'attachment; filename="Previous results on mask detection in videos.xls"'
    wb = xlwt.Workbook(encoding='utf-8')
    ws = wb.add_sheet('Sheet 1')
    row_num = 0
    font_style = xlwt.XFStyle()
    font_style.font.bold = True
    columns = ['Sr no.', 'Location', 'Date and Time', 'No. of Violations']
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
    
