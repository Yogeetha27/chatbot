# app.py (Flask application)

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import model
import os
from model import predict_skin_disease  # Import predict_skin_disease function from your model module

app = Flask(__name__)

# Specify the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Ensure that 'model' and 'classes' are accessible
        try:
            # Assuming model and classes are defined in model module
            result = predict_skin_disease(file_path, model.model, model.classes)
            return f'<h1 style="color: blue; text-align: center;">Predicted Skin Disease: {result}</h1>'
        except FileNotFoundError:
            return f'Error: File not found - {file_path}'
        except Exception as e:
            return f'Error predicting skin disease: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)
