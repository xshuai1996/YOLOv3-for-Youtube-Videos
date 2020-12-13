import pafy
import cv2
import numpy as np
import torch
from darknet import Darknet
import argparse
import torchvision.transforms as transforms
from utils import load_class_names, pad_to_square, resize_image, boxes_filter, rescale_boxes


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./yolov3.cfg", help="path to configuration file")
    parser.add_argument("--weights_path", type=str, default="./yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="./coco.names", help="path to class name file")
    parser.add_argument("--conf_thre", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--NMS_thre", type=float, default=0.4, help="iou threshold for Non-Maximum Suppression")
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup the model
    model = Darknet(cfgfile=args.config_path)
    model.load_weights(weight_path=args.weights_path)
    model.eval()

    # prepare the online video
    classes = load_class_names(class_name_path=args.class_path)
    colors = [tuple(255 * np.random.rand(3)) for i in range(len(classes))]
    img_size = int(model.net_params['height'])
    video_url = "https://youtu.be/jjlBnrzSGjc?t=10"

    vPafy = pafy.new(video_url)
    play = vPafy.getbest(preftype="any")
    cap = cv2.VideoCapture(play.url)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # save video
    save_video = cv2.VideoWriter("./result.avi", cv2.VideoWriter_fourcc(*'MJPG'), 15, (frame_width, frame_height))

    # while video not end
    while True:
        ret, img = cap.read()
        if ret:
            frame = transforms.ToTensor()(img)
            frame, _ = pad_to_square(frame, 0)
            # resize the frame in accordance with the size in configuration file
            frame = resize_image(frame, size=img_size).reshape((1, 3, img_size, img_size)).to(device)

            # get the detection result
            with torch.no_grad():
                result = model(frame)
                result = boxes_filter(result, args.conf_thre, args.NMS_thre)[0]

            detections = rescale_boxes(result, img_size, img.shape[:2])
            for x1, y1, x2, y2, conf, _, cls_pred in detections:
                tl = (x1, y1)
                br = (x2, y2)
                label = classes[int(cls_pred)]
                # draw rectangles and tags for boxes
                img = cv2.rectangle(img, tl, br, colors[int(cls_pred)], 2)
                img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            save_video.write(img)
            cv2.imshow(video_url, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    result.release()
    torch.cuda.empty_cache()
    cv2.destroyAllWindows()
