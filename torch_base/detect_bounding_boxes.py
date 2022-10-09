import numpy as np
import cv2
import torch
import glob as glob
import sys
import os
# sys.path.append('/media/karthikragunath/Personal-Data/carla_6/RL_CARLA/torch_base')
# sys.path.append('/media/karthikragunath/Personal-Data/carla_6/RL_CARLA')
sys.path.append(os.path.join(os.getcwd(), 'torch_base'))
sys.path.append(os.getcwd())
from faster_rcnn_model import create_model
import matplotlib.pyplot as plt

# set the computation device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# load the model and the trained weights
model = create_model(num_classes=5).to(device)
print("1" * 50)
model.load_state_dict(torch.load(
    os.path.join(os.getcwd(), 'torch_base/pretrained_model/model10.pth'), map_location=device
))
# model.load_state_dict(torch.load(
#     '/media/karthikragunath/Personal-Data/carla_6/RL_CARLA/torch_base/pretrained_model/model10.pth', map_location=device
# ))
model.eval()
# classes: 0 index is reserved for background
CLASSES = [
    '0', '1', '2', '3', '4'
]

# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.8

__all__ = ['DetectBoundingBox']
directory_path = "/media/karthikragunath/Personal-Data/carla_6/RL_CARLA/carla_rgb_sensor_flow_detected"
bounding_box_directory_path = "/media/karthikragunath/Personal-Data/carla_6/RL_CARLA/bounding_box_outputs"

class DetectBoundingBox:
    def __init__(self, image, image_name):
        self.image = image
        self.image_name = image_name

    def detect_bounding_boxes(self):
        # print("Image Name (Self):", self.image_name)
        # image = cv2.imread(directory_path + "/" + self.image_name)
        image = self.image
        if not image.any():
            print(directory_path, self.image_name, "EMPTY IMAGE OBJ")
        orig_image = image.copy()
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float64)
        # convert to tensor
        # image = torch.tensor(image, dtype=torch.float).cuda()
        image = torch.tensor(image, dtype=torch.float)
        image = image.to(device)
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image)

        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                cv2.rectangle(orig_image,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              (0, 0, 255), 2)
            # plt.imshow(orig_image)
            # plt.savefig(bounding_box_directory_path + "/" + self.image_name)
            # print("Bounding Box Path:", bounding_box_directory_path, self.image_name)
        # print("Image {image_name} done...".format(image_name=self.image_name))
        return orig_image