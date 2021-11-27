import argparse
import io
import os
from PIL import Image

import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms as T
from torch import nn
import numpy as np
import zipfile
from flask import Flask, render_template, request, redirect

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
					classes = {0: 'Princess', 1: "Just a tiger"}
					area = (xmin, ymin, xmax, ymax)
					img = img.crop(area)
					predictions = list()
					for model__ in efb3_models:
						tensor = torch.unsqueeze(transforms['val'](img), 0)
						predictions.append(np.array(model__(tensor).detach()))
					preds = np.array(predictions).mean(axis=0)
					is_vasya = np.argmax(
							nn.functional.softmax(torch.FloatTensor(preds), dim=-1)).item()
					title = classes[is_vasya]
				else:
					title = name
				img.save('static/crop.jpg', format="JPEG")
			results.render()  # updates results.imgs with boxes and labels
			for img in results.imgs:
				img_base64 = Image.fromarray(img)
				img_base64.save("static/image0.jpg", format="JPEG")
			return render_template("classify.html", image="static/image0.jpg", title=title)
		elif file.filename.strip().split('.')[1] == 'zip':
			#file_like_object = file.stream._file
			archive = zipfile.ZipFile(file.stream._file)
			file_names = archive.namelist()
			file_names = [file_name for file_name in file_names if file_name.endswith(".jpg") or file_name.endswith(".jpeg")]


	return render_template("index.html")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
	parser.add_argument("--port", default=5000, type=int, help="port number")
	args = parser.parse_args()

	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	model = torch.hub.load('ultralytics/yolov5', 'custom', path='./super_models/best_mixed_medium.pt', force_reload=True)
	model.eval()
	tiger_vasya_models = list()
	files = os.listdir(ENSAMBLE_BASEDIR)
	for n, file in enumerate(files):
		effnet_b3 = EfficientNet.from_pretrained('efficientnet-b3', num_classes=2)
		state_dict = torch.load(ENSAMBLE_BASEDIR + 'model_ensemble_' + str(n))
		effnet_b3.load_state_dict(state_dict)
		effnet_b3.eval()
		tiger_vasya_models.append(effnet_b3)

	efb3_models = tiger_vasya_models
	print(len(efb3_models))
	app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
