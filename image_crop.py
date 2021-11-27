import argparse
import io
import os
from PIL import Image

import torch
SOURCE_PATH_0 = "../Downloads/ХАК/Принцесса_400/"
DIST_PATH_0 = "../Downloads/ХАК/princes_croped/"
SOURCE_PATH_1 = "../Downloads/ХАК/Train_3500/3 класса/Тигры/"
DIST_PATH_1 = "../Downloads/ХАК/tiger_croped/"

model = torch.hub.load('ultralytics/yolov5', 'custom', path='../Downloads/best.pt', force_reload=True)
model.eval()
model = model.to('cpu')

files = os.listdir(SOURCE_PATH_0)
for file in files:
	if not file.endswith('jpg'):
		continue
	img = Image.open(SOURCE_PATH_0 + file)
	results = model(img, size=640)
	data = results.pandas().xyxy[0].to_json(orient="records") #json render response
	if str(data) == '[]':
		continue
	xmax = int(results.pandas().xyxy[0].iloc[0]['xmax'])
	xmin = int(results.pandas().xyxy[0].iloc[0]['xmin'])
	ymax = int(results.pandas().xyxy[0].iloc[0]['ymax'])
	ymin = int(results.pandas().xyxy[0].iloc[0]['ymin'])
	conf = float(results.pandas().xyxy[0].iloc[0]['confidence'])
	cls = int(results.pandas().xyxy[0].iloc[0]['class'])
	name = str(results.pandas().xyxy[0].iloc[0]['name'])
	print(file)
	area = (xmin, ymin, xmax, ymax)
	print(area)
	img = img.crop(area)
	img.save(DIST_PATH_0 + file, format="JPEG")
