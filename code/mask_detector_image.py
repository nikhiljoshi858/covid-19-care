from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import imutils

path_to_image = 'D:/Django_Projects/temp/nomask.jpg'

image = cv2.imread(path_to_image)

prototxtPath = 'D:/Django_Projects/temp/models/deploy.prototxt'
weightsPath = 'D:/Django_Projects/temp/models/resnet.caffemodel'


net = cv2.dnn.readNet(prototxtPath, weightsPath)

orig = image.copy()
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

print("[INFO] computing face detections...")
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

        # Nikhil (Local)
        model = load_model('D:/Django_Projects/temp/models/temp')
        print('[INFO] Loading Saved model...')
        mask, without_mask = model.predict(face)[0]
        
        # Srivatsan (Docker)
        # headers = {'content-type': 'applications/json'}
        # data = json.dumps({'signature_type': 'serving_default', 'instances': face.tolist()})
        # json_response = requests.post('http://localhost:8501/v1/models/temp:predict', data=data, headers=headers)
        # predictions = json.loads(json_response.text)
        # mask, without_mask = predictions['predictions'][0]
        
        
        if mask > without_mask:
            label = 'Mask'
            color = (0, 255, 0)
        else:
            label = 'No Mask'
            color = (0, 0, 255)

        print('[INFO] Predictions got...')

        label = '{}: {:.2f}%'.format(label, max(mask, without_mask) * 100)

        print('[INFO] Writing results...')

        cv2.putText(image, label, (startx, starty-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startx, starty), (endx, endy), color, 2)


cv2.imshow('Output', image)
cv2.waitKey(0)
