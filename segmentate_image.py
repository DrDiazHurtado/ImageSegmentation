import cv2
import numpy as np

file_path = '< YOUR_PATH >'

# Random dogs image ;-D
image_path = file_path+'dog.jpg'  
image = cv2.imread(image_path)

# Load YOLO model and configuration
net = cv2.dnn.readNet(file_path+'yolov3.weights', file_path+'yolov3.cfg')

# Load class names)
with open(file_path+"coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Get image dimensions
height, width, _ = image.shape

# Preprocess the image and pass it through the network
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outs = net.forward(net.getUnconnectedOutLayersNames())

# Define  thresholds
conf_threshold = 0.5
nms_threshold = 0.4

# Bounding box coordinates, class IDs and confidences
boxes = []
class_ids = []
confidences = []

# Detections
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            class_ids.append(class_id)
            confidences.append(float(confidence))

# Overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
print(indices)
# Draw on the image
for i in indices:
    box = boxes[i]
    x, y, w, h = box
    label = str(classes[class_ids[i]])
    color = (0, 255, 0)  # BGR color for the bounding box (green)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display images
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
