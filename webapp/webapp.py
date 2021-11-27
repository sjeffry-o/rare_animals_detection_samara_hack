import argparse
import io
from io import StringIO
import os
from PIL import Image
#import Image
import copy
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms as T
from torch import nn
import numpy as np
import zipfile
from flask import Flask, render_template, request, redirect
import os.path
from os import path
from PIL import Image
import imghdr
app = Flask(__name__)

ENSAMBLE_BASEDIR = './super_models/ensamble_effnet_b3/'


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

classes = {0: 'Princess', 1: "Tiger"}



@app.route("/", methods=["GET", "POST"])
def predict():
	if request.method == "POST":
		if "file" not in request.files:
			return redirect(request.url)
		file = request.files["file"]
		if not file:
			return
		if file.filename.strip().split('.')[1] == 'jpg':
			img_bytes = file.read()
			img = Image.open(io.BytesIO(img_bytes))
			# img = img.crop((0, 0, img.size[0], img.size[1] - 200))
			results = model(img, size=640)
			data = results.pandas().xyxy[0].to_json(orient="records") #json render response
			title = ''
			if str(data) != '[]':
				xmax = int(results.pandas().xyxy[0].iloc[0]['xmax'])
				xmin = int(results.pandas().xyxy[0].iloc[0]['xmin'])
				ymax = int(results.pandas().xyxy[0].iloc[0]['ymax'])
				ymin = int(results.pandas().xyxy[0].iloc[0]['ymin'])
				conf = float(results.pandas().xyxy[0].iloc[0]['confidence'])
				cls = int(results.pandas().xyxy[0].iloc[0]['class'])
				name = str(results.pandas().xyxy[0].iloc[0]['name'])
				if name == 'Tiger':
					area = (xmin, ymin, xmax, ymax)
					img = img.crop(area)
					predictions = list()
					for model__ in efb3_models:
						tensor = torch.unsqueeze(transforms['val'](img), 0)
						predictions.append(np.array(model__(tensor).detach()))
					preds = np.array(predictions).mean(axis=0)
					is_princess = np.argmax(
							nn.functional.softmax(torch.FloatTensor(preds), dim=-1)).item()
					title = classes[is_princess]
				else:
					title = name
				if title not in ('Princess', 'Tiger', 'Leopard'):
					title = 'Other'
				img.save('static/crop.jpg', format="JPEG")
			results.render()  # updates results.imgs with boxes and labels
			for img in results.imgs:
				img_base64 = Image.fromarray(img)
				img_base64.save("static/image0.jpg", format="JPEG")
			return render_template("classify.html", image="static/image0.jpg", title=title, filename=file.filename)
		elif file.filename.strip().split('.')[1] == 'zip':
			if not path.exists(f"predicted"):
				os.mkdir(f"predicted")
			predicts_csv = open("./predicted/predicts.csv", "w+")
			predicts_csv.write("img,label\n")
			archive = zipfile.ZipFile(file.stream._file)
			file_names = archive.namelist()
			file_names = [file_name for file_name in file_names if (file_name.endswith(".jpg") or file_name.endswith(".jpeg")) and "__MACOSX" not in file_name]
			print(file_names)
			for file_ in file_names:
				print(file_)
				try:
					data = archive.read(file_)
					dataEnc = io.BytesIO(data)
					img = Image.open(dataEnc)
				except:
					with open("predicted/log", 'a') as log:
						log.write(f"Can't open {file_}\n")
					continue
				results = model(img, size=640)
				data = results.pandas().xyxy[0].to_json(orient="records") #json render response
				if str(data) != '[]':
					xmax = int(results.pandas().xyxy[0].iloc[0]['xmax'])
					xmin = int(results.pandas().xyxy[0].iloc[0]['xmin'])
					ymax = int(results.pandas().xyxy[0].iloc[0]['ymax'])
					ymin = int(results.pandas().xyxy[0].iloc[0]['ymin'])
					conf = float(results.pandas().xyxy[0].iloc[0]['confidence'])
					cls = int(results.pandas().xyxy[0].iloc[0]['class'])
					name = str(results.pandas().xyxy[0].iloc[0]['name'])
					if name == 'Tiger':
						area = (xmin, ymin, xmax, ymax)
						img = img.crop(area)
						predictions = list()
						for model__ in efb3_models:
							tensor = torch.unsqueeze(transforms['val'](img), 0)
							predictions.append(np.array(model__(tensor).detach()))
						preds = np.array(predictions).mean(axis=0)
						is_princess = np.argmax(
								nn.functional.softmax(torch.FloatTensor(preds), dim=-1)).item()
						title = classes[is_princess]
					else:
						title = name
					predicts_csv.write(f"{file_.split('/')[-1]},{title}\n")
					if not path.exists(f"predicted/{title}"):
						os.mkdir(f"predicted/{title}")
					results.render()  # updates results.imgs with boxes and labels
					for img in results.imgs:
						img_base64 = Image.fromarray(img)
						img_base64.save(f"predicted/{title}/{file_.split('/')[-1]}", format="JPEG")
			predicts_csv.close()
			predicts = list()
			with open("./predicted/predicts.csv", 'r') as preds:
				for line in preds:
					predicts.append(line.strip())
			return render_template("predict.html", predicts=predicts)
	return render_template("index.html")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
	parser.add_argument("--port", default=5000, type=int, help="port number")
	args = parser.parse_args()

	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	model = torch.hub.load('ultralytics/yolov5',
			'custom',
			path='./super_models/best_given_others.pt',
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
	app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
