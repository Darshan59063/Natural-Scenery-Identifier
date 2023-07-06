from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
import urllib.request
import os
from werkzeug.utils import secure_filename
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np


app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024



labels = {0 : "Buildings",
1 : "Forest",
2 : "Glacier",
3 : "Mountain",
4 : "Sea",
5 : "Street",
6:  "Others" }

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index2.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        print(file)
        filename = secure_filename(file.filename)
        # print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded and displayed below')

        # Load an image from a file
        image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image = image.resize((224, 224))

        image_array = np.array(image)
        # print(image_array)
        image_array = image_array / 255.0
        # print(image_array.shape)
        image_array = np.expand_dims(image_array, axis=0)
        # print(image_array.shape)

        cnn_model = load_model('static/util/intel.h5')
        predictions = cnn_model.predict(image_array)
        # print(predictions)
        predicted_label = labels[np.argmax(predictions[0])]
        pred_prob_percentage = ((predictions[0][np.argmax(predictions[0])]*100).round(decimals=2))
        # print(pred_prob_percentage)
        if(pred_prob_percentage<50):
            pred_prob_percentage = 100 - pred_prob_percentage
            predicted_label = labels[6]
        return render_template('index2.html', filename=filename, label_pred = predicted_label, pred_prob_percentage= pred_prob_percentage)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    # print(redirect(url_for('static', filename='uploads/' + filename)))
    # # redirect(url_for('static', filename='uploads/' + filename), code=301)
    # return 1

@app.route('/styles/<filename>')
def styles(filename):
    return send_from_directory('templates', filename)

@app.route('/clear')
def clear():
    return  redirect(url_for('/'))

if __name__ == "__main__":
    app.run(debug=True)