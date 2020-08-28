#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import numpy as np
import requests
from copy import deepcopy
from io import BytesIO
from PIL import Image

def image_from_url(url):
    response = requests.get(url)
    return cv2.cvtColor(np.array(Image.open(BytesIO(response.content))), cv2.COLOR_RGB2BGR)

def view_image(cv2_img):
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes, COLORS):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def setup_classes(file_classes):
    with open(file_classes, 'r') as f:
      classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return classes, colors

def setup_net(file_weights, file_config):
    return cv2.dnn.readNet(file_weights, file_config)

    
def recognize(input_image, file_classes=None, file_weights=None, file_config=None, classes=None, COLORS=None, net=None, copy_image=True, imsize=None):
    image = deepcopy(input_image) if copy_image else input_image

    if imsize is None:
        Width = image.shape[1]
        Height = image.shape[0]
    else:
        Width, Height = imsize
    scale = 0.00392


    if classes is None:
        classes, COLORS = setup_classes(file_classes)


    if net is None:
        net = setup_net(file_weights, file_config)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
      for detection in out:
          scores = detection[5:]
          class_id = np.argmax(scores)
          confidence = scores[class_id]
          if confidence > 0.5:
              center_x = int(detection[0] * Width)
              center_y = int(detection[1] * Height)
              w = int(detection[2] * Width)
              h = int(detection[3] * Height)
              x = center_x - w / 2
              y = center_y - h / 2
              class_ids.append(class_id)
              confidences.append(float(confidence))
              boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
      i = i[0]
      box = boxes[i]
      x = box[0]
      y = box[1]
      w = box[2]
      h = box[3]
      draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), classes, COLORS)
  
    return image

def live_recognition(file_weights, file_config, file_classes, camera_id=0):
    net = setup_net(file_weights, file_config)
    classes, COLORS = setup_classes(file_classes)

    cap = cv2.VideoCapture(camera_id)

    first_time = True
    while(True):
        _, frame = cap.read()
        if first_time:
            imsize = (frame.shape[1], frame.shape[0])
            first_time = False

        img = recognize(frame, classes=classes, COLORS=COLORS, net=net, copy_image=False, imsize=imsize)
        cv2.imshow("Hit 'q' to exit", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    # net = setup_net(,"yolov3.cfg")
    # classes,COLORS = setup_classes("yolov3.txt")

    live_recognition("../../Downloads/yolov3.weights", "yolov3.cfg", "yolov3.txt")
