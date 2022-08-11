# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class Yolov5DeepSortTracker:
    def __init__(self, input_source, output_callback):
        with torch.no_grad():
            self.init(input_source, output_callback)

    def init(self, input_source, output_callback):
        yolo_model = 'weights/cone.pt'
        deep_sort_model = 'osnet_x0_25'
        device_id = ''
        self.augment = True
        self.show_vid = True
        self.half = False
        self.dnn = True
        self.config_deepsort = 'deep_sort/configs/deep_sort.yaml'
        self.conf_thres = 0.5
        self.iou_thres = 0.5
        self.classes = None
        self.agnostic_nms = True
        self.max_det = 1000
        self.imgsz = [640, 640]
        self.output_callback = output_callback

        source = input_source

        # Initialize
        self.device = select_device(device_id)

        # Directories
        if type(yolo_model) is str:  # single yolo model
            exp_name = yolo_model.split(".")[0]
        elif type(yolo_model) is list and len(yolo_model) == 1:  # single models after --yolo_model
            exp_name = yolo_model[0].split(".")[0]
        else:  # multiple models after --yolo_model
            exp_name = "ensemble"
        exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]
        # save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run if project name exists
        # (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        self.model = DetectMultiBackend(yolo_model, device=self.device, dnn=self.dnn)
        stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Half
        self.half &= pt and self.device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            self.model.model.half() if self.half else self.model.model.float()

        # Set Dataloader
        vid_path, vid_writer = None, None

        # Dataloader
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.dataset = LoadStreams(source, img_size=self.imgsz, stride=stride, auto=pt)
        self.nr_sources = len(self.dataset)

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(self.config_deepsort)

        # Create as many trackers as there are video sources
        self.deepsort_list = []
        for i in range(self.nr_sources):
            self.deepsort_list.append(
                DeepSort(
                    deep_sort_model,
                    self.device,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE,
                    n_init=cfg.DEEPSORT.N_INIT,
                    nn_budget=cfg.DEEPSORT.NN_BUDGET,
                )
            )

        # Get class names and colors
        names = self.model.module.names \
            if hasattr(self.model, 'module') else self.model.names

        self.model.warmup(
            imgsz=(1 if pt else self.nr_sources, 3, *self.imgsz)
        )  # warmup

    def run(self):
        # Run tracking
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        outputs = [None] * self.nr_sources
        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(self.dataset):
            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            pred = self.model(
                im,
                augment=self.augment
            )
            t3 = time_sync()
            dt[1] += t3 - t2

            # Apply NMS
            pred = non_max_suppression(
                pred,
                self.conf_thres,
                self.iou_thres,
                self.classes,
                self.agnostic_nms,
                max_det=self.max_det
            )
            dt[2] += time_sync() - t3

            ret = None

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                im0 = im0s[i].copy()
                annotator = Annotator(im0, line_width=2, pil=not ascii)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        im.shape[2:],
                        det[:, :4],
                        im0.shape
                    ).round()

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to deepsort
                    t4 = time_sync()
                    outputs[i] = self.deepsort_list[i].update(
                        xywhs.cpu(),
                        confs.cpu(),
                        clss.cpu(),
                        im0
                    )
                    t5 = time_sync()
                    dt[3] += t5 - t4

                    # draw boxes for visualization
                    if len(outputs[i]) > 0:
                        for j, (output) in enumerate(outputs[i]):

                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            conf = output[6]

                            c = int(cls)  # integer class
                            label = f'{id:0.0f} {self.names[c]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(c, True))

                    self.output_callback(outputs)
                    LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
                else:
                    self.deepsort_list[i].increment_ages()
                    LOGGER.info('No detections')

                # Stream results
                im0 = annotator.result()
                if self.show_vid:
                    cv2.imshow(str(Path(path[i])), im0)
                    cv2.waitKey(1)  # 1 millisecond

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
            per image at shape {(1, 3, *self.imgsz)}' % t)
