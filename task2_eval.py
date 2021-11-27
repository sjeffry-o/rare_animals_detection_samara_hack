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


effb3_size = 300
effb5_size = 456
effb7_size = 600
___size = effb3_size

transforms = {
	'train': T.Compose([
		T.Resize((___size, ___size)),
		T.RandomHorizontalFlip(),
		T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0),
		T.ToTensor(),
		T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': T.Compose([
		T.Resize((___size, ___size)),
		T.ToTensor(),
		T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
}


model = torch.hub.load('ultralytics/yolov5',
			'custom',
			path='./webapp/super_models/best.pt',
			force_reload=True)
model = model.to(device)
model.eval()

tiger_princess_models = list()
files = os.listdir(ENSAMBLE_BASEDIR)
base_effnet_b3 = EfficientNet.from_pretrained('efficientnet-b3', num_classes=2)
for n, file in enumerate(files):
	effnet_b3 = copy.deepcopy(base_effnet_b3)
	state_dict = torch.load(ENSAMBLE_BASEDIR + 'model_ensemble_' + str(n))
	effnet_b3.load_state_dict(state_dict)
	effnet_b3.eval()
	effnet_b3 = effnet_b3.to(device)
	tiger_princess_models.append(effnet_b3)
efb3_models = tiger_princess_models

classes = {0: 'Princess', 1: "Tiger"}
convert = {0: 1, 1:0}
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
				xmax = int(results.pandas().xyxy[0].iloc[0]['xmax'])
				xmin = int(results.pandas().xyxy[0].iloc[0]['xmin'])
				ymax = int(results.pandas().xyxy[0].iloc[0]['ymax'])
				ymin = int(results.pandas().xyxy[0].iloc[0]['ymin'])
				conf = float(results.pandas().xyxy[0].iloc[0]['confidence'])
				cls = int(results.pandas().xyxy[0].iloc[0]['class'])
				name = str(results.pandas().xyxy[0].iloc[0]['name'])
				area = (xmin, ymin, xmax, ymax)
				img = img.crop(area)
				predictions = list()
				for model__ in efb3_models:
					tensor = torch.unsqueeze(transforms['val'](img), 0)
					predictions.append(np.array(model__(tensor).detach()))
				preds = np.array(predictions).mean(axis=0)
				is_princess = np.argmax(nn.functional.softmax(torch.FloatTensor(preds), dim=-1)).item()
				is_princess = convert[is_princess]
				output.write(f"{file},{is_princess}\n")
			else:
				output.write(f"{file},1\n")
			tq.update(1)




if __name__ == "__main__":
	path = sys.argv[1]
	main(path)
