from os.path import exists
from urllib3 import PoolManager
import torch
import torch.nn.functional as F
import numpy as np


def parse_config(config_path):
    """
    Check if configuration file exist, and download if not. Parse the configuration file and return a list.
    :param config_path: path of configuration file
    :return config: a list containing dicts, each dict is a config block
    """
    # download if not exist
    if not exists(config_path):
        print("File yolov3.cfg not exist. Downloading now.")
        config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
        r = PoolManager().request('GET', config_url)
        with open(config_path, 'wb') as f:
            f.write(r.data)
        r.release_conn()
        print("File yolov3.cfg downloaded.")

    # preprocess config file
    config_file = open(config_path, 'r')
    lines = config_file.read().split('\n')
    filtered_lines = []
    for line in lines:
        # remove empty lines, space lines, and comment lines, and remove space in lines
        if len(line) == 0 or line.isspace() is True or line[0] == '#':
            continue
        filtered_lines.append(line.replace(' ', ''))

    # parse config
    config = []
    for line in filtered_lines:
        if line[0] == '[':      # new block starts
            config.append({})
            block = config[-1]
            block['type'] = line[1:-1]
        else:
            key, value = line.split('=')
            block[key] = value

    # by default the height and width are supposed to be equal
    if config[0]['width'] != config[0]['height']:
        print("By default 'height' and 'width' in configuration should be same, "
              "and image will be resize to height x height in this case.")
    if int(config[0]['width']) != 416 or int(config[0]['height']) != 416:
        print("To match with the paper both 'height' and 'width' are recommended to be 416. Current values are {}, {}."
              .format(int(config[0]['width']), int(config[0]['height'])))
    return config


def download_pretrained_weights(weight_path):
    # download if not exist
    if not exists(weight_path):
        print("File yolov3.weights not exist. Downloading now.")
        weight_url = "https://pjreddie.com/media/files/yolov3.weights"
        r = PoolManager().request('GET', weight_url)
        with open(weight_path, 'wb') as f:
            f.write(r.data)
        r.release_conn()
        print("File yolov3.weights downloaded.")
    return weight_path


def load_class_names(class_name_path):
    # download if not exist
    if not exists(class_name_path):
        print("File coco.names not exist. Downloading now.")
        class_name_url = "https://raw.githubusercontent.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/master/data/coco.names"
        r = PoolManager().request('GET', class_name_url)
        with open(class_name_path, 'wb') as f:
            f.write(r.data)
        r.release_conn()
        print("File coco.names downloaded.")

    f = open(class_name_path, "r")
    class_names = f.read().split("\n")[:-1]
    return class_names


def boxes_filter(img_box_pred, conf_thre, NMS_thre=0.4):
    """
    Remove boxes with too low confidence and perform Non-Maximum Suppression.
    :param img_box_pred: the prediction for all boxes (i.e. output of Darknet)
    :param conf_thre: confidence threshold
    :param NMS_thre: NMS threshold
    :return: qualified boxes in the form (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # update the (x, y, w, h) to the position of corners
    # tensor.new creates a new tensor of specified shape with the same data type of boxes
    box_corner = img_box_pred.new(torch.Size((img_box_pred.shape[0], img_box_pred.shape[1], 4)))
    box_corner[:, :, 0] = (img_box_pred[:, :, 0] - img_box_pred[:, :, 2] / 2)
    box_corner[:, :, 1] = (img_box_pred[:, :, 1] - img_box_pred[:, :, 3] / 2)
    box_corner[:, :, 2] = (img_box_pred[:, :, 0] + img_box_pred[:, :, 2] / 2)
    box_corner[:, :, 3] = (img_box_pred[:, :, 1] + img_box_pred[:, :, 3] / 2)
    img_box_pred[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(img_box_pred.shape[0])]
    for img_ind, img_pred in enumerate(img_box_pred):
        # discard boxes with too low confidence
        img_pred = img_pred[img_pred[:, 4] >= conf_thre]
        # If no box remain go directly to next image
        if not img_pred.shape[0]:
            continue

        # sort all boxes to put the boxes with larger score first
        # score = object confidence x best class confidence
        score = img_pred[:, 4] * img_pred[:, 5:].max(1)[0]
        img_pred = img_pred[(-score).argsort()]

        # perform non-maximum suppression
        max_conf_score, max_conf_class = img_pred[:, 5:].max(1, keepdim=True)
        boxes = torch.cat((img_pred[:, :5], max_conf_score.float(), max_conf_class.float()), 1)
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = bbox_iou(boxes[0:1, :4], boxes[:, :4]) > NMS_thre
            label_match = boxes[0, -1] == boxes[:, -1]

            # invalid if overlap too much and belong to same class
            invalid = large_overlap & label_match
            # change the coordinates of the final box to a weighted sum form of all those overlapping boxes
            weights_of_invalid_boxes = boxes[invalid, 4:5]
            boxes[0, :4] = (weights_of_invalid_boxes * boxes[invalid, :4]).sum(0) / weights_of_invalid_boxes.sum()
            keep_boxes += [boxes[0]]
            # discard the overlapping boxes
            boxes = boxes[~invalid]
        if keep_boxes:
            output[img_ind] = torch.stack(keep_boxes, dim=0)

    return output


def bbox_iou(box1, box2):
    """Returns the IoU of two bounding boxes. box2 could also be multiple boxes."""
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def pad_to_square(img, pad_value):
    """pad the given image to square with given value"""
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad


def resize_image(img, size):
    image = F.interpolate(img.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def rescale_boxes(boxes, current_dim, original_shape):
    """Rescales bounding boxes to the actual size (i.e. eliminate the effect of pad_to_square and resize_image)"""
    if boxes is None:
        return []
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes



