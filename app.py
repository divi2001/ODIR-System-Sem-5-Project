from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import model_from_json
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

name = 'ODIR_ResNet'

json_file = open('./'+name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./"+name+".h5")
print("Loaded model from disk")

labels = pd.read_csv("labels.txt", sep="\n").values


@app.route('/')
def index():
	return render_template("index.html", data="hey")


@app.route("/prediction", methods=["POST"])
def prediction():


	left = request.files['left']
	left.save("left.jpg")

	img = cv2.imread("left.jpg")
	img = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = np.array(img)
	left=[]
	left.append(img)


	right = request.files['right']
	right.save("right.jpg")

	img = cv2.imread("right.jpg")
	img = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = np.array(img)
	right=[]
	right.append(img)

	X1 = np.array(left)
	X2 = np.array(right)

	yhat = loaded_model.predict([X1, X2])
	a = yhat
	aa = np.argmax(a)

	if aa == 0:
		aa="Normal Eye"
	elif aa == 1:
		aa="Diabetes"
	elif aa == 2:
		aa="Glaucomna"
	elif aa == 3:
		aa="Catarct"
	elif aa == 4:
		aa="Macular Degeneration"
	elif aa == 5:
		print="Hypertension"
	elif aa == 6:
		aa="Myopia"
	elif aa == 7:
		aa="Other Diseases"




	return render_template("prediction.html", data=aa)


if __name__ == "__main__":
	app.run(debug=True)
