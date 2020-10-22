from django.shortcuts import render
from django.contrib.auth.decorators import login_required
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


# Create your views here.

MIN_CONF = 0.3
NMS_THRESH = 0.3
MIN_DISTANCE_IMAGE = 100


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

			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	return results


def previous_results_view(request):
    context = {}
    objects = Previous_Social.objects.all().order_by('-timestamp')
    context['previous'] = objects
    return render(request, 'social/previous.html', context=context)


@login_required(redirect_field_name='/social/')
def homepage_view(request):
    return render(request, 'social/home.html')


@login_required(redirect_field_name='/social/image')
def image_view(request):
    if request.method == "POST":
        img = cv2.imdecode(np.fromstring(request.FILES['files'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        labelsPath = 'D:/Django_Projects/temp/models/coco.names'
        LABELS = open(labelsPath).read().strip().split("\n")

        weightsPath = 'D:/Django_Projects/temp/models/yolov3.weights'
        configPath = 'D:/Django_Projects/temp/models/yolov3.cfg'

        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        results = detect_people(img, net, ln, personIdx=LABELS.index("person"))

        violate = set()

        if len(results) >= 2:
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < MIN_DISTANCE_IMAGE:
                        violate.add(i)
                        violate.add(j)

        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
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

        row = Previous_Social(result=len(violate), category='image')
        row.save()

        context = {}
        context['image'] = uri
        context['number'] = len(violate)

        return render(request, 'social/image_output.html', context)


    return render(request, 'social/image.html')


@login_required(redirect_field_name='/social/video')
def video_view(request):
    return render(request, 'social/video.html')


@login_required(redirect_field_name='/social/webcam')
def webcam_view(request):
    labelsPath = '/home/nikhil/Desktop/models/coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")

    weightsPath = '/home/nikhil/Desktop/models/yolov3.weights'
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


    row = Previous_Social(result=len(violate), category='video')
    row.save()


    return render(request, 'social/webcam.html')

