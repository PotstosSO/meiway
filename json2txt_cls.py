import os
import cv2
import glob
import json
import numpy as np
from tqdm import tqdm


# 将labelme_json标注转yolo_txt
def convert(size, box):
    """
    convert [xmin, xmax, ymin, ymax] to [x_centre, y_centre, w, h]
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


if __name__ == "__main__":

    class_names = []

    # json文件夹路径
    json_dir = r"/mnt/d/Datasets/silyworm_segm/(4)"
    # 转化后txt的保存路径
    txt_dir = r"/mnt/d/Datasets/silyworm_segm/label"

    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir, exist_ok=True)
    json_pths = glob.glob(json_dir + "/*.json")

    for json_pth in tqdm(json_pths,  desc='Processing'):
        f1 = open(json_pth, "r")
        json_data = json.load(f1)

        img_pth = os.path.join(json_dir, json_pth.replace("json", "jpg"))
        img = cv2.imread(img_pth)
        h, w = img.shape[:2]

        tag = os.path.basename(json_pth)
        out_file = open(os.path.join(txt_dir, tag.replace("json", "txt")), "w")
        label_infos = json_data["shapes"]

        for label_info in label_infos:
            label = label_info["label"]
            points = label_info["points"]
            if len(points) >= 3:
                points = np.array(points)
                xmin, xmax = max(0, min(np.unique(points[:, 0]))), min(w, max(np.unique(points[:, 0])))
                ymin, ymax = max(0, min(np.unique(points[:, 1]))), min(h, max(np.unique(points[:, 1])))
            elif len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
            else:
                continue
            bbox = [xmin, xmax, ymin, ymax]
            bbox_ = convert((w,h), bbox)
            if label not in class_names:
                class_names.append(label)
            cls_id = class_names.index(label)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bbox_]) + '\n')
    with open(txt_dir + 'classes.txt', 'w') as f:
        for i in class_names:
            f.write(i + '\n')
