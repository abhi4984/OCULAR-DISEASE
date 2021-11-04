from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

dic = {0: 'Normal', 1: 'Diabetic Retinopathy'}

model = load_model('my_model')

model.make_predict_function()


def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i) / 255.0
    i = i.reshape(1, 224, 224, 3)
    # p = model.predict_classes(i)
    y_pre = model.predict(i)
    y_pred = [np.argmax(j) for j in y_pre]
    return dic[y_pred[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/about")
def about_page():
    return "Welcome to Gravitas AI!!!!"


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)

    return render_template("index.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    # app.debug = True
    app.run(debug=False)
