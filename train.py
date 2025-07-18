import warnings
import os
import torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    torch.cuda.empty_cache()
    model = YOLO('ultralytics/cfg/models/RS.yaml')
    # model.load('yolov10n.pt') # loading pretrain weights
    model.train(data='',
                cache=False,
                imgsz=640,
                epochs=500,
                batch=16,
                close_mosaic=0,
                workers=8,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='',
                name='',
                )
