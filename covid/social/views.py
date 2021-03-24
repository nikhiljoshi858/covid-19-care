# Project Name: 	COVID-19 Care: Face Mask and Social Distancing Detection using Deep Learning
# Author List: 		Nikhil Joshi
# Filename: 		views.py
# Functions: 		previous_results_view(request), detect_people(frame, net, ln, personIdx), get_boundary_points_video(frame)
#                   get_mouse_points_video(event, x, y, flags, param), get_boundary_points(image), get_mouse_points(event, x, y, flags, param)
#                   get_social_distancing_view(image, results, threshold_distance, matrix), order_points(pts), four_point_transform(image, pts)
#                   get_social_distancing_view_video(frame, results, copied_frame, threshold_distance, sd_output_video, matrix), get_birds_eye_view(image, mapping)
#                   get_birds_eye_view_video(frame, mapping, bdv_output_video), homepage_view(request), image_view(request), video_view(request)
#                   webcam_view(request), previous_results_image_csv_view(request), previous_results_video_csv_view(request), previous_results_image_xlsx_view(request)
#                   previous_results_video_xlsx_view(request)
# Global Variables:	location, MIN_CONF, NMS_THRESH, mouse_pts, mouse_pts_video, points, mapping, 
#                   violate, centroids, image, copied_image, frame


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


# The following piece of code fetches the user location by using the Geolookup library.
# We need the API key of the library and using that we can fetch the city, region and 
# country of the user. This data is stored in the previous results database table.
global location
geo_lookup = GeoLookup('776da34f4f37c2fb8f3ad306cc615bff')
location = geo_lookup.get_own_location()
location = location['city'] + ', ' + location['region_name'] + ', ' + location['country_name']


# Function Name:	previous_results_view
# Input:		    request: HTTP request
# Output:		    The previous results page is rendered
# Logic:		    Retrieve the previous results from the database and order them by timestamp.
#                   Pass them in the template and render the template.
# Example Call:		Called by Django when the URL for this function is triggered
def previous_results_view(request):
    context = {}
    objects = Previous_Social.objects.all().order_by('timestamp')
    context['previous'] = objects
    return render(request, 'social/previous.html', context=context)


# Defining some thresholds and global variables
MIN_CONF = 0.5
NMS_THRESH = 0.5
mouse_pts = []
mouse_pts_video = []
points = []
mapping = {}
violate = set()
centroids = []


# Loading the YOLO object detector
labelsPath = 'D:/Django_Projects/temp/models/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = 'D:/Django_Projects/temp/models/yolov3.weights'
configPath = 'D:/Django_Projects/temp/models/yolov3.cfg'


# Function Name:	detect_people
# Input:		    frame: the frame/image on which social distancing detection is to be used
#                   ln: the layer names of the YOLO object detector
#                   net: the YOLO object detector
#                   personIdx: the index of the class of object to be detected. Default value is 0 (for persons)
# Output:		    A list of bounding boxes, and class probabilities of all persons detected in the input image/frame
# Logic:		    Get the height and width of the input image/frame. Run one iterattion of forward propogation
#                   on the YOLO object detector. This will detect all the possible objects in the image/frame.
#                   Get the class id of the object detected, the confidence of the detection. If the confidence
#                   is greater than a threshold value (like 50% or 0.5) and the class id is 0 (for a person) get
#                   the bounding box coordinates, calculate the centroid and append all these values to the respective
#                   lists. Then apply non-max suppression on the detections to suppress the bounding boxes with a low
#                   confidence. Form a list of these results after non-max suppression and return this list.
# Example Call:		get_boundary_points_video(frame)
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



# Function Name:	get_boundary_points_video
# Input:		    image: the image on which the bird's eye view is to be applied
# Output:		    The list of mouse points clicked for the bird's eye view and the threshold distance calcualtion
# Logic:		    The function is called by the get_boundary_points function while recording the mouse points clicked.
#                   For the first 4 points mark the points red and for the last 2 points mark the points blue. Also draw
#                   a line between successive points so that the 4 first points clicked form a quadrilateral. The bird's
#                   eye view is formed inside this quadrilateral only.
# Example Call:		get_boundary_points_video(frame)
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


# Function Name:	get_mouse_points_video
# Input:		    event: the event which took place, like clicking the mouse
#                   x: x-coordinate of the mouse point
#                   y: y-coordinate of the mouse point
#                   flags, param: required parameters
# Output:		    The list of mouse points clicked for the bird's eye view and the threshold distance calcualtion
# Logic:		    The function is called by the get_boundary_points function while recording the mouse points clicked.
#                   For the first 4 points mark the points red and for the last 2 points mark the points blue. Also draw
#                   a line between successive points so that the 4 first points clicked form a quadrilateral. The bird's
#                   eye view is formed inside this quadrilateral only.
# Example Call:		Called by the get_boundary_points function
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


# Function Name:	get_boundary_points
# Input:		    image: the image on which the bird's eye view is to be applied
# Output:		    The list of mouse points clicked for the bird's eye view and the threshold distance calcualtion
# Logic:		    The function is called by the get_boundary_points function while recording the mouse points clicked.
#                   For the first 4 points mark the points red and for the last 2 points mark the points blue. Also draw
#                   a line between successive points so that the 4 first points clicked form a quadrilateral. The bird's
#                   eye view is formed inside this quadrilateral only.
# Example Call:		get_boundary_points_video(image)
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


# Function Name:	get_mouse_points
# Input:		    event: the event which took place, like clicking the mouse
#                   x: x-coordinate of the mouse point
#                   y: y-coordinate of the mouse point
#                   flags, param: required parameters
# Output:		    The list of mouse points clicked for the bird's eye view and the threshold distance calcualtion
# Logic:		    The function is called by the get_boundary_points function while recording the mouse points clicked.
#                   For the first 4 points mark the points red and for the last 2 points mark the points blue. Also draw
#                   a line between successive points so that the 4 first points clicked form a quadrilateral. The bird's
#                   eye view is formed inside this quadrilateral only.
# Example Call:		Called by the get_boundary_points function
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


# Function Name:	get_social_distancing_view
# Input:		    image: the image on which social distancing detection is to be applied
#                   results: the list of the persons detected for that frame and the bounding boxes 
#                   along with the probabilitiy
#                   threshold_distance: the threshold distance required between 2 people
#                   matrix: the perspective transform matrix required to map the original centroid 
#                   of the bounding box to the bird'e eye view point so that the bird's eye view
#                   looks similar to the top view of the frame
# Output:		    The social distancing detection output for the uploaded image by the user
# Logic:		    Iterate through the results, record the start and end coordinates of the bounding box
#                   and the centroids as well. If the length of the results array is more
#                   than 1, or if more than 1 persons are detected in the frame, find the centroids list 
#                   using the Euclidean distance formula between every pair of centroids. Iterate through
#                   the list using 2 arrays and if the distance between any pair of centroids is less than
#                   the threshold value, add both the persons to the violate set. Since a set is being used,
#                   the same person cannot be added multiple times. Then, iterate through the results once more and 
#                   find the transformed coordinates of the centroids using the perspective transform matrix.
#                   Then, if that particular person is present in the violate set, then the risk factor of that
#                   person is 'risk' and the bounding box will be marked red. Otherwise the risk factor will be
#                   'norisk' and the bounding box will be marked green. Finally display the bounding boxes, the no.
#                   of violations on the copied image. Return the image.
# Example Call:		get_social_distancing_view_video(frame, results, copied_frame, threshold_distance, sd_output_video, matrix)
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


# Function Name:	order_points
# Input:		    pts: the list of boundary points on which bird's eye view is applied
# Output:		    The ordered points as required to apply for the perspective transformation.
# Logic:		    To apply the perspective transformation for the bird'e eye view, the points 
#                   must be ordered in the following order: top left, top right, bottom right, 
#                   bottom left. This function orders them in this order, given any 4 points. It 
#                   calculates the sum and differences of the original points (sum and differences 
#                   of the respective x and y coordinates). It then forms another list, which represents 
#                   the ordered points from these 2 values. Finally the ordered points are returned.
# Example Call:		order_points(pts)
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


# Function Name:	four_point_transform
# Input:		    image: the image on which the perspective transformation is to be applied
#                   pts: the list of boundary points on which bird's eye view is applied
# Output:		    The ordered points as required to apply for the perspective transformation.
# Logic:		    To apply the perspective transformation for the bird'e eye view, the points 
#                   must be ordered in the following order: top left, top right, bottom right, 
#                   bottom left. This function orders them in this order, given any 4 points. It 
#                   calculates the sum and differences of the original points (sum and differences 
#                   of the respective x and y coordinates). It then forms another list, which represents 
#                   the ordered points from these 2 values. Finally the ordered points are returned.
# Example Call:		four_point_transform(image, pts)
def four_point_transform(image, pts):
    rect = order_points(pts)
    h, w = image.shape[:2]
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (w, h))
    return matrix, warped


# Function Name:	get_social_distancing_view_video
# Input:		    frame: the frame on which social distancing detection is to be applied
#                   results: the list of the persons detected for that frame and the bounding boxes 
#                   along with the probabilitiy
#                   copied_frame: copy of the original frame
#                   threshold_distance: the threshold distance required between 2 people
#                   sd_output_video: the video object to which the frames are written
#                   matrix: the perspective transform matrix required to map the original centroid 
#                   of the bounding box to the bird'e eye view point so that the bird's eye view
#                   looks similar to the top view of the frame
# Output:		    The social distancing detection output video for the uploaded video by the user
# Logic:		    Initialise an empty set violate(). Then if the length of the results array is more
#                   than 1, or if more than 1 persons are detected in the frame, find the centroids list 
#                   using the Euclidean distance formula between every pair of centroids. Iterate through
#                   the list using 2 arrays and if the distance between any pair of centroids is less than
#                   the threshold value, add both the persons to the violate set. Since a set is being used,
#                   the same person cannot be added multiple times. Then, iterate through the results once 
#                   again, record the start and end coordinates of the bounding box and the centroids as well.
#                   Find the transformed coordinates of the centroids using the perspective transform matrix.
#                   Then, if that particular person is present in the violate set, then the risk factor of that
#                   person is 'risk' and the bounding box will be marked red. Otherwise the risk factor will be
#                   'norisk' and the bounding box will be marked green. Finally display the bounding boxes, the no.
#                   of violations and write the frame to the output video object. Return the video object.
# Example Call:		get_social_distancing_view_video(frame, results, copied_frame, threshold_distance, sd_output_video, matrix)
def get_social_distancing_view_video(frame, results, copied_frame, threshold_distance, sd_output_video, matrix):
    violate = set()
    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean").tolist()
        for i in range(len(D[0])):
            for j in range(i+1, len(D[1])):
                if D[i][j] < 100: #Threshold distance
                    violate.add(i)
                    violate.add(j)

    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
        x = startX + ((endX - startX) // 2)
        y = endX
        px = int((matrix[0][0]*x + matrix[0][1]*y + matrix[0][2]) / (matrix[2][0]*x + matrix[2][1]*y + matrix[2][2]))
        py = int((matrix[1][0]*x + matrix[1][1]*y + matrix[1][2]) / (matrix[2][0]*x + matrix[2][1]*y + matrix[2][2]))
        mapping[(i, px, py)] = 'norisk'
        if i in violate:
            color = (0, 0, 255)
            mapping[(i, px, py)] = 'risk'

        cv2.rectangle(copied_frame, (startX, startY), (endX, endY), color, 2)
    
    text = "Social Distancing Violations: {}".format(len(violate))
    cv2.putText(copied_frame, text, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.85, (0, 0, 255), 3)
    sd_output_video.write(copied_frame)
    

# Function Name:	get_birds_eye_view
# Input:		    image: the image on which the bird's eye view is to be applied
#                   mapping: a dictionary which maps the persons whether they are at risk
#                   to give the appropriate color to the bouding
# Output:		    The bird's eye view of the uploaded image
# Logic:		    Construct a white image whose size is same as the original image. 
#                   Then iterate through the mapping dictionary, and while iterating if a person's
#                   risk factor is 'risk' then mark that centroid red in color. Otherwise mark it 
#                   green. Repeat the process for every centroid. Finally, return this image to the callee
# Example Call:		get_birds_eye_view(frame, mapping, bdv_output_video)
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
    


# Function Name:	get_birds_eye_view_video
# Input:		    frame: the frame on which the bird's eye view is to be applied
#                   mapping: a dictionary which maps the persons whether they are at risk
#                   to give the appropriate color to the bouding
#                   bdv_output_video: video object on which the output frames are written
# Output:		    The bird's eye view video
# Logic:		    Construct a white image whose size is same as the original frame size. 
#                   Then iterate through the mapping dictionary, and while iterating if a person's
#                   risk factor is 'risk' then mark that centroid red in color. Otherwise mark it 
#                   green. Repeat the process for every centroid. Finally write this image to the video output object
# Example Call:		get_birds_eye_view_video(frame, mapping, bdv_output_video)
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


# Function Name:	homepage_view
# Input:		    HTTP request
# Output:		    The homepage is displayed
# Logic:		    Render the HTML and display the same.
# Example Call:		Called by Django when the particular URL is called by the user
@login_required(redirect_field_name='/social/')
def homepage_view(request):
    return render(request, 'social/home.html')


# Function Name:	image_view
# Input:		    HTTP request
# Output:		    The image uploaded by the user and the output when the social distancing module
#                   is applied on that image. The bird's eye view of the image is also displayed.
# Logic:		    Read the COCO model and load the labels in a text file. Read the image by decoding
#                   the file uploaded in the earlier view. Here, the user is required to select 6 points
#                   first by clicking the mouse. The first 4 points represent the coordinates of the 
#                   boundaries for the bird's eye view. The perspective transformation function will 
#                   be applied on these points to get the bird's eye view. The next 2 points represent the 
#                   threshold distance required for maintaining social distancing i.e. 6 feet. It can be 
#                   assumed that the average height of a person is 5.5 to 6 feet and hence, those points 
#                   could be selected for this purpose. Then, for the entire image, firstly detect the
#                   people found in the frame using the YOLO weights. Find the bounding box coordinates for every person 
#                   detected and calculate the centroid of the box by finding the mean of the x and y coordinates 
#                   respectively. Then iterate through the results using 2 loops and find the Euclidean distance
#                   of every pair of centroids. If this distance is found less than the threshold distance measured 
#                   earlier, then add the placeholder of these persons to the violate set, The persons added 
#                   in this set are the violations calculated for that image and so the length of this set will
#                   be the number of violations for the image. Display the no. of violations, and the bounding box for the image.
#                   The bounding box will be red for violators and green for non-violators. Finally send the image by encoding it
#                   as a base64 uri object and display the same in the HTML file.
# Example Call:		Called by Django when the particular URL is called by the user
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


# Function Name:	webcam_view
# Input:		    HTTP request
# Output:		    The video uploaded by the user and the output when the social distancing module
#                   is applied on that video. The bird's eye view of the video is also displayed.
# Logic:		    Read the COCO model and load the labels in a text file. Initialize 2 video objects, 
#                   one each for the social distancing output and the bird's eye view. Open the 
#                   video using the opencv library function cv2.VideoCapture(). For the first frame, 
#                   the user is required to select 6 points by clicking the mouse. The first 4 points represent the 
#                   coordinates of the boundaries for the bird's eye view. The perspective transformation function will 
#                   be applied on these points to get the bird's eye view. The next 2 points represent the threshold distance 
#                   required for maintaining social distancing i.e. 6 feet. It can be assumed that the average height of a 
#                   person is 5.5 to 6 feet and hence, those points could be selected for this purpose. For every successive 
#                   frame captured, firstly detect the people found in the frame using the YOLO weights. Find the bounding box 
#                   coordinates for every person detected and calculate the centroid of the box by finding the mean of the 
#                   x and y coordinates respectively. Then iterate through the results using 2 loops and find the Euclidean 
#                   distance of every pair of centroids. If this distance is found less than the threshold distance measured 
#                   earlier, then add the placeholder of these persons to the violate set The persons added in this set are the 
#                   violations calculated for that particular frame and so the length of this set will be the number of 
#                   violations for that frame. Display the no. of violations, and the bounding box for every frame.
#                   The bounding box will be red for violators and green for non-violators. Finally display the frame. 
#                   The entire process is repeated for every frame
# Example Call:		Called by Django when the particular URL is called by the user
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
            ordered_points = order_points(mouse_pts_video[0:4])
            matrix, warped = four_point_transform(frame, ordered_points)
            results = detect_people(frame, net, ln, personIdx=0)
            violate = set()
            if len(results) >= 2:
                get_social_distancing_view_video(frame, results, copied_frame, threshold_distance, sd_video_output, matrix)
                get_birds_eye_view_video(warped, mapping, bdv_video_output)
        
        return render(request, 'social/video_output.html')

    return render(request, 'social/video.html')


# Function Name:	webcam_view
# Input:		    HTTP request
# Output:		    The live webcam feed of the user and the output when the social distancing module
#                   is applied on the webcam feed. If any person is found violating the social distancing norms,
#                   an alarm will be sounded
# Logic:		    Read the COCO model and load the labels in a text file. Start the webcam feed using the opencv library function
#                   cv2.VideoCapture(0). For the first frame, the user is required to select 6 points by clicking the mouse.
#                   The first 4 points represent the coordinates of the boundaries for the bird's eye view. The perspective 
#                   transformation function will be applied on these points to get the bird's eye view. The next 2 points 
#                   represent the threshold distance required for maintaining social distancing i.e. 6 feet. It can be assumed 
#                   that the average height of a person is 5.5 to 6 feet and hence, those points could be selected for this purpose. 
#                   For every successive frame captured, firstly detect the people found in the frame using the YOLO weights. 
#                   Find the bounding box coordinates for every person detected and calculate the centroid of the box by finding 
#                   the mean of the x and y coordinates respectively. Then iterate through the results using 2 loops and 
#                   find the Euclidean distance of every pair of centroids. If this distance is found less than the threshold
#                   distance measured earlier, then add the placeholder of these persons to the violate set. The persons added in 
#                   this set are the violations measured for that particular frame and so the length of this set will be the 
#                   number of violations for that frame. Display the no. of violations, and the bounding box for every frame. The 
#                   bounding box will be red for violators and green for non-violators. Finally display the frame. The entire process
#                   is repeated for every frame
# Example Call:		Called by Django when the particular URL is called by the user
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
                    if D[i, j] < 100:
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


# Function Name:	previous_results_image_csv_view
# Input:		    HTTP request
# Output:		    A CSV file comprising of previous results when the social distancing module
#                   is applied on images
# Logic:		    Obtain the previous results records from the database, construct a workbook using xlwt library
# 		            and fill the workbook row wise while fetching data from the database. Display the link to the file
#                   as a downloadable link so that the user can download the file
# Example Call:		Called by Django when the particular URL is called by the user
def previous_results_image_csv_view(request):
    rows = Previous_Social.objects.filter(category='image')
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="Previous results on Social Distancing detection in images.csv"'
    writer = csv.writer(response)
    writer.writerow(['Sr no.', 'Location', 'Date and Time', 'No. of Violations'])
    for row in rows:
        writer.writerow([row.id, row.location, row.timestamp, row.result])
    return response


# Function Name:	previous_results_video_csv_view
# Input:		    HTTP request
# Output:		    A CSV file comprising of previous results when the social distancing module
#                   is applied on videos
# Logic:		    Obtain the previous results records from the database, construct a workbook using xlwt library
# 		            and fill the workbook row wise while fetching data from the database. Display the link to the file
#                   as a downloadable link so that the user can download the file
# Example Call:		Called by Django when the particular URL is called by the user
def previous_results_video_csv_view(request):
    rows = Previous_Social.objects.filter(category='video')
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="Previous results on Social Distancing detection in videos.csv"'
    writer = csv.writer(response)
    writer.writerow(['Sr no.', 'Location', 'Date and Time', 'No. of Violations'])
    for row in rows:
        writer.writerow([row.id, row.location, row.timestamp, row.result])
    return response
    

# Function Name:	previous_results_image_xlsx_view
# Input:		    HTTP request
# Output:		    An excel file comprising of previous results when the social distancing module
#                   is applied on images
# Logic:		    Obtain the previous results records from the database, construct a workbook using xlwt library
# 		            and fill the workbook row wise while fetching data from the database. Display the link to the file
#                   as a downloadable link so that the user can download the file
# Example Call:		Called by Django when the particular URL is called by the user
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
    

# Function Name:	previous_results_video_xlsx_view
# Input:		    HTTP request
# Output:		    An excel file comprising of previous results when the social distancing module
#                   is applied on videos
# Logic:		    Obtain the previous results records from the database, construct a workbook using xlwt library
# 		            and fill the workbook row wise while fetching data from the database. Display the link to the file
#                   as a downloadable link so that the user can download the file
# Example Call:		Called by Django when the particular URL is called by the user
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