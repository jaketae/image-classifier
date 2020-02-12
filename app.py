from flask import Flask, render_template, request, redirect
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import base64
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 12})


MODEL = load_model('mobilenet.h5')
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


def check_extension(file_name):
    return '.' in file_name and \
           file_name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_prediction(predictions):
    pred_array = predictions[0]
    label = [prediction[1].lower() for prediction in pred_array]
    estimate = [prediction[2] for prediction in pred_array]
    return label, estimate


def create_plot(label, estimate):
    output = BytesIO()
    fig = Figure(figsize=(16,10), dpi=300, tight_layout=True)
    axis = fig.add_subplot(1, 1, 1)
    axis.bar(label, estimate, color='#007bff')
    FigureCanvas(fig).print_png(output)
    image_string = "data:image/png;base64,"
    image_string += base64.b64encode(output.getvalue()).decode('utf8')
    return image_string


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    raw_image = request.files["image"]
    if not check_extension(raw_image.filename):
        return redirect('/error')
    else:
        image = Image.open(BytesIO(raw_image.read()))
        processed_image = preprocess_image(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        prediction_array = MODEL.predict(processed_image)
        prediction = decode_predictions(prediction_array)
        label, estimate = parse_prediction(prediction)
        image_string = create_plot(label, estimate)
        return render_template('predict.html', image=image_string, label=label)
    


@app.route('/error')
def error():
    return render_template('error.html')

   

if __name__ == "__main__":
    app.run(debug=True)