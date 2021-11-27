import argparse
import io
import os
from PIL import Image

import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict():
	if request.method == "POST":
		if "file" not in request.files:
			return redirect(request.url)
		file = request.files["file"]
		if not file:
			return
		img_bytes = file.read()
		img = Image.open(io.BytesIO(img_bytes))
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
			area = (xmin, ymin, xmax, ymax)
			img = img.crop(area)

	# ADD RUNNING TIME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			img.save('static/crop.jpg', format="JPEG")
		results.render()  # updates results.imgs with boxes and labels
		for img in results.imgs:
			img_base64 = Image.fromarray(img)
			img_base64.save("static/image0.jpg", format="JPEG")
		return redirect("static/image0.jpg")

	return render_template("index.html")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
	parser.add_argument("--port", default=5000, type=int, help="port number")
	args = parser.parse_args()
	
	model = torch.hub.load('ultralytics/yolov5', 'custom', path='./super_models/best.pt', force_reload=True)
	#model = torch.hub.load(
	#	"ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True)  # force_reload = recache latest code
	model.eval()
	app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
