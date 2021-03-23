from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.core.files import File
from django.conf import settings
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

global location
geo_lookup = GeoLookup('776da34f4f37c2fb8f3ad306cc615bff')
location = geo_lookup.get_own_location()
location = location['city'] + ', ' + location['region_name'] + ', ' + location['country_name']


def previous_results_view(request):
    context = {}
    objects = Previous_Social.objects.all().order_by('timestamp')
    context['previous'] = objects
    return render(request, 'social/previous.html', context=context)


MIN_CONF = 0.5
NMS_THRESH = 0.5
mouse_pts = []
mouse_pts_video = []
points = []
mapping = {}
violate = set()
centroids = []


labelsPath = 'D:/Django_Projects/temp/models/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = 'D:/Django_Projects/temp/models/yolov3.weights'
configPath = 'D:/Django_Projects/temp/models/yolov3.cfg'


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




def get_boundary_points_video(frame):
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", get_mouse_points_video)
    while True:
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            frame = frame.copy()
        elif key == ord("c"):
            break
        if len(mouse_pts_video) <= 4:
            frame = cv2.putText(frame, "Enter the boundaries of the Bird's Eye View", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if len(mouse_pts_video) > 3 and len(mouse_pts_video) <= 6:
            frame = cv2.putText(frame, "Now, Enter the Threshold distance", (50,75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if len(mouse_pts_video) == 6:
            cv2.destroyWindow("frame")
            break

def get_mouse_points_video(event, x, y, flags, param):
    global mouse_pts_video
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts_video) < 4:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), 10)
        else:
            cv2.circle(frame, (x, y), 5, (255, 0, 0), 10)
            
        if len(mouse_pts_video) >= 1 and len(mouse_pts_video) <= 3:
            cv2.line(frame, (x, y), (mouse_pts_video[len(mouse_pts_video)-1][0], mouse_pts_video[len(mouse_pts_video)-1][1]), (70, 70, 70), 2)
            if len(mouse_pts_video) == 3:
                cv2.line(frame, (x, y), (mouse_pts_video[0][0], mouse_pts_video[0][1]), (70, 70, 70), 2)

        if "mouse_pts_video" not in globals():
            mouse_pts_video = []
        mouse_pts_video.append([x, y])


def get_boundary_points(image):
    cv2.namedWindow("image")
    cropping = True
    if cropping:
        cv2.setMouseCallback("image", get_mouse_points)
    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break
        if len(mouse_pts) <= 4:
            image = cv2.putText(image, "Enter the boundaries of the Bird's Eye View", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if len(mouse_pts) > 3 and len(mouse_pts) <= 6:
            image = cv2.putText(image, "Now, Enter the Threshold distance", (50,75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if len(mouse_pts) == 6:
            cv2.destroyWindow("image")
            cropping = False
        if not cropping:
            break


def get_mouse_points(event, x, y, flags, param):
    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        else:
            cv2.circle(image, (x, y), 5, (255, 0, 0), 10)
            
        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts)-1][0], mouse_pts[len(mouse_pts)-1][1]), (70, 70, 70), 2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)
        
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append([x, y])



def get_social_distancing_view(image, results, threshold_distance, matrix):
    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
        x = startX + ((endX - startX) // 2)
        y = endY
        px = int((matrix[0][0]*x + matrix[0][1]*y + matrix[0][2]) // (matrix[2][0]*x + matrix[2][1]*y + matrix[2][2]))
        py = int((matrix[1][0]*x + matrix[1][1]*y + matrix[1][2]) // (matrix[2][0]*x + matrix[2][1]*y + matrix[2][2]))
        centroids.append((px, py))

    if len(results) >= 2:
        D = dist.cdist(centroids, centroids, metric="euclidean").tolist()
        for i in range(len(D[0])):
            for j in range(i+1, len(D[0])):
                if D[i][j] < threshold_distance:
                    violate.add(i)
                    violate.add(j)

    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
        mapping[(i, centroids[i][0], centroids[i][1])] = 'norisk'

        if i in violate:
            mapping[(i, px, py)] = 'risk'
            color = (0, 0, 255)

        cv2.rectangle(copied_image, (startX, startY), (endX, endY), color, 2)

    text = "Social Distancing Violations: {}".format(len(violate))
    cv2.putText(copied_image, text, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.85, (0, 0, 255), 3)
    return copied_image

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    pts = np.asarray(pts)
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    h, w = image.shape[:2]
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (w, h))
    return matrix, warped



def get_social_distancing_view_video(frame, results, copied_frame, threshold_distance, sd_output_video, matrix):
    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
        x = startX + ((endX - startX) // 2)
        y = endY
        px = int((matrix[0][0]*x + matrix[0][1]*y + matrix[0][2]) // (matrix[2][0]*x + matrix[2][1]*y + matrix[2][2]))
        py = int((matrix[1][0]*x + matrix[1][1]*y + matrix[1][2]) // (matrix[2][0]*x + matrix[2][1]*y + matrix[2][2]))
        centroids.append((px, py))

    if len(results) >= 2:
        D = dist.cdist(centroids, centroids, metric="euclidean").tolist()
        for i in range(len(D[0])):
            for j in range(i+1, len(D[0])):
                if D[i][j] < threshold_distance:
                    violate.add(i)
                    violate.add(j)

    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
        mapping[(i, centroids[i][0], centroids[i][1])] = 'norisk'

        if i in violate:
            mapping[(i, px, py)] = 'risk'
            color = (0, 0, 255)

        cv2.rectangle(copied_frame, (startX, startY), (endX, endY), color, 2)
    
    text = "Social Distancing Violations: {}".format(len(violate))
    cv2.putText(copied_frame, text, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.85, (0, 0, 255), 3)
    sd_output_video.write(copied_frame)
    


def get_birds_eye_view(image, mapping):
    h = image.shape[0]
    w = image.shape[1]

    red = (0, 0, 255)
    green = (0, 255, 0)
    white = (200, 200, 200)

    birds_eye_view = np.zeros((h, w, 3), np.uint8)
    birds_eye_view[:] = white

    for (i, px, py), risk_factor in mapping.items():
        if risk_factor == 'risk':
            cv2.circle(birds_eye_view, (px, py), 5, red, -1)
        else:
            cv2.circle(birds_eye_view, (px, py), 5, green, -1)
    
    return birds_eye_view
    

def get_birds_eye_view_video(frame, mapping, bdv_output_video):
    red = (0, 0, 255)
    green = (0, 255, 0)
    white = (200, 200, 200)

    bdv = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]), np.uint8)
    bdv[:] = white
    for (i, px, py), risk_factor in mapping.items():
        if risk_factor == 'risk':
            cv2.circle(bdv, (px, py), 10, red, -1)
        else:
            cv2.circle(bdv, (px, py), 10, green, -1)
    
    bdv_output_video.write(bdv)


@login_required(redirect_field_name='/social/')
def homepage_view(request):
    return render(request, 'social/home.html')


@login_required(redirect_field_name='/social/image')
def image_view(request):
    if request.method == "POST":
        global image, copied_image
        image = cv2.imdecode(np.fromstring(request.FILES['files'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        copied_image = image.copy()
        
        labelsPath = 'D:/Django_Projects/temp/models/coco.names'
        LABELS = open(labelsPath).read().strip().split("\n")

        weightsPath = 'D:/Django_Projects/temp/models/yolov3.weights'
        configPath = 'D:/Django_Projects/temp/models/yolov3.cfg'

        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        get_boundary_points_video(image)
        results = detect_people(image, net, ln, personIdx=0)

        threshold_distance = ((mouse_pts[-1][0] - mouse_pts[-2][0]) ** 2 + (mouse_pts[-1][1] - mouse_pts[-2][1]) ** 2) ** 0.5

        ordered_points = order_points(mouse_pts[0:4])
        matrix, warped = four_point_transform(image, ordered_points)

        sd_view = get_social_distancing_view(image, results, threshold_distance, matrix)
        bdv = get_birds_eye_view(warped, mapping) 

        uri1 = b64encode(cv2.imencode('.jpg', sd_view)[1]).decode()
        uri1 = "data:%s;base64,%s" % ("image/jpeg", uri1)
        uri2 = b64encode(cv2.imencode('.jpg', bdv)[1]).decode()
        uri2 = "data:%s;base64,%s" % ("image/jpeg", uri2)
        

        row = Previous_Social(result=len(violate), category='image', location=location)
        row.save()

        context = {}
        context['sd_view'] = uri1
        context['birds_eye_view'] = uri2
        context['number'] = len(violate)

        return render(request, 'social/image_output.html', context)

    return render(request, 'social/image.html')


@login_required(redirect_field_name='/social/video')
def video_view(request):
    if request.method == 'POST':
        video = request.FILES['video']
        v = Video(video=video)
        v.save()

        v = Video.objects.last()
        video = cv2.VideoCapture(v.video.path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sd_video_output = cv2.VideoWriter(settings.MEDIA_ROOT+'/sd_output_video.mp4',fourcc, fps, (width, height))
        bdv_video_output = cv2.VideoWriter(settings.MEDIA_ROOT+'/bdv_output_video.mp4',fourcc, fps, (width, height))

        labelsPath = 'D:/Django_Projects/temp/models/coco.names'
        LABELS = open(labelsPath).read().strip().split("\n")
        weightsPath = 'D:/Django_Projects/temp/models/yolov3.weights'
        configPath = 'D:/Django_Projects/temp/models/yolov3.cfg'
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        cropping = True
        global frame
        while True:
            grabbed, frame = video.read()
            if not grabbed:
                break
            copied_frame = frame.copy()
            if cropping:
                get_boundary_points_video(frame)
                threshold_distance = ((mouse_pts_video[-1][0] - mouse_pts_video[-2][0]) ** 2 + (mouse_pts_video[-1][1] - mouse_pts_video[-2][1]) ** 2) ** 0.5
                cropping = False
            results = detect_people(frame, net, ln, personIdx=0)
            ordered_points = order_points(mouse_pts_video[0:4])
            matrix, warped = four_point_transform(frame, ordered_points)
            get_social_distancing_view_video(frame, results, copied_frame, threshold_distance, sd_video_output, matrix)
            get_birds_eye_view_video(warped, mapping, bdv_video_output)
        
        return render(request, 'social/video_output.html')

    return render(request, 'social/video.html')


@login_required(redirect_field_name='/social/webcam')
def webcam_view(request):
    # labelsPath = 'D:/Django_Projects/temp/models/coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")

    # weightsPath = 'E:\Django_Projects/temp/models/yolov3.weights'
    # configPath = '/home/nikhil/Desktop/models/yolov3.cfg'

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
    
