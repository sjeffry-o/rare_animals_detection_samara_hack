import argparse
import io
import os
from PIL import Image
import copy
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms as T
from torch import nn
import numpy as np
import zipfile
import os.path
from os import path
import sys
ENSAMBLE_BASEDIR = './webapp/super_models/ensamble_effnet_b3/'
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5',
			'custom',
			path='./webapp/super_models/best.pt',
			force_reload=True)
model = model.to(device)
model.eval()
names = {"Tiger": '1', 'Leopard': '2', 'Other': '3'}

def main(path):
	output = open('./labels.csv', 'w+')
	output.write("id,class\n")
	files = os.listdir(path)
	files = [file for file in files if file.endswith('.jpg') or file.endswith('.jpeg')]
	with tqdm(total=len(files)) as tq:
		for file in files:
			img = Image.open('/' + path.strip('/') + '/' +  file)
			results = model(img, size=640)
			data = results.pandas().xyxy[0].to_json(orient="records")
			if data != '[]':
				results.render()  # updates results.imgs with boxes and labels
				name = str(results.pandas().xyxy[0].iloc[0]['name'])
				label = names.get(name)
				output.write(f"{file},{label if label else '3'}\n")
			else:
				output.write(f"{file},3\n")
			tq.update(1)




if __name__ == "__main__":
	path = sys.argv[1]
	main(path)
