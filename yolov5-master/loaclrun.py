import torch
import sys
import cv2
import numpy as np
from pathlib import Path
import os
import torch
import torchvision

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results."""
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence threshold

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        output[xi] = x[i[:max_det]]
    return output

def xywh2xyxy(x):
    """Convert [x,y,w,h] to [x1,y1,x2,y2]"""
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
# 添加项目根目录到Python路径
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# 导入YOLOv5模块
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.augmentations import letterbox
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode


# ===== 参数设置 =====
class Opt:
    def __init__(self):
        self.weights = [str(ROOT / 'runs/train/exp7/weights/best.pt')]

        # 设置推理图片目录
        self.try_dir = Path('C:/Users/21435/Desktop/railway2/try')
        os.makedirs(self.try_dir, exist_ok=True)
        self.source = str(self.try_dir)

        # 如果目录为空，自动添加一张示例图
        if not any(self.try_dir.glob('*.[jpJP][pnPN]*[gG]')):  # 匹配jpg/png/jpeg图像
            sample_img = ROOT / 'data/images/bus.jpg'
            if sample_img.exists():
                import shutil
                shutil.copy(sample_img, self.try_dir / 'bus.jpg')
                print(f"✅ 已添加示例图片: {sample_img.name}")

        # 其他推理参数
        self.data = None
        self.imgsz = [640]
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.device = 'cpu'
        self.view_img = True
        self.save_txt = False
        self.save_conf = False
        self.save_crop = False
        self.nosave = False
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.visualize = False
        self.update = False
        self.project = str(ROOT / 'runs/detect')
        self.name = 'exp'
        self.exist_ok = True
        self.line_thickness = 3
        self.hide_labels = False
        self.hide_conf = False
        self.half = False
        self.dnn = False
        self.vid_stride = 1


opt = Opt()


@smart_inference_mode()
def run():
    print(f"📂 正在加载图片目录: {opt.source}")
    device = select_device(opt.device)
    model = DetectMultiBackend(opt.weights[0], device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = opt.imgsz if len(opt.imgsz) == 2 else [opt.imgsz[0], opt.imgsz[0]]

    dataset = LoadImages(opt.source, img_size=imgsz, stride=stride, auto=pt)

    for path, im, im0s, vid_cap, s in dataset:
        im = letterbox(im0s, imgsz, stride=stride, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]  # BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(device)
        im = im.half() if opt.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = model(im, augment=opt.augment, visualize=opt.visualize)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)

        for det in pred:
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=opt.line_thickness, example=str(names))

            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            if opt.view_img:
                cv2.imshow(Path(path).name, annotator.result())
                cv2.waitKey(0)

    print("✅ 推理完成！")

if __name__ == "__main__":
    run()