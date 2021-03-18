import numpy as np
import cv2
from scipy.spatial import distance as dist
import imutils

MIN_CONF = 0.5
NMS_THRESH = 0.5
mouse_pts = []
points = []
mapping = {}

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



def get_boundary_points(image):
    cv2.namedWindow("image")
    cropping = True
    if cropping:
        cv2.setMouseCallback("image", get_mouse_points)
    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            image = clone.copy()
        elif key == ord("c"):
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


def get_social_distancing_view(image, results):
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
    cv2.imshow('image', copied_image)
    cv2.waitKey(0)


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


def get_transformed_points(boxes, prespective_transform):
    bottom_points = []
    for box in boxes:
        pnts = np.array([[[int(box[0]+(box[2]*0.5)),int(box[1]+box[3])]]] , dtype="float32")
        bd_pnt = cv2.perspectiveTransform(pnts, prespective_transform)[0][0]
        pnt = [int(bd_pnt[0]), int(bd_pnt[1])]
        bottom_points.append(pnt)
    return bottom_points


def get_birds_eye_view(image, mapping):
    h = image.shape[0]
    w = image.shape[1]

    red = (0, 0, 255)
    green = (0, 255, 0)
    white = (200, 200, 200)

    blank_image = np.zeros((h, w, 3), np.uint8)
    blank_image[:] = white

    for (i, px, py), risk_factor in mapping.items():
        if risk_factor == 'risk':
            cv2.circle(blank_image, (px, py), 5, red, -1)
        else:
            cv2.circle(blank_image, (px, py), 5, green, -1)
    
    cv2.imshow("Bird's Eye View", blank_image)
    cv2.waitKey(0)


path = 'D:/Django_Projects/temp/ffsd.jpg'
image = cv2.imread(path)
get_boundary_points(image)
centroids = []
copied_image = cv2.imread(path)
violate = set()
labelsPath = 'D:/Django_Projects/temp/models/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = 'D:/Django_Projects/temp/models/yolov3.weights'
configPath = 'D:/Django_Projects/temp/models/yolov3.cfg'
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
results = detect_people(image, net, ln, personIdx=0)

threshold_distance = ((mouse_pts[-1][0] - mouse_pts[-2][0]) ** 2 + (mouse_pts[-1][1] - mouse_pts[-2][1]) ** 2) ** 0.5
# print('Threshold distance : ',threshold_distance)

ordered_points = order_points(mouse_pts[0:4])
matrix, warped = four_point_transform(image, ordered_points)
# print('Matrix: ', matrix)
# cv2.imshow("Perspective O/P", warped)
# cv2.waitKey(0)
get_social_distancing_view(image, results)
get_birds_eye_view(warped, mapping)