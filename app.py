import os
import numpy as np
from flask import Flask, render_template, request, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Charger ton modèle une fois
model = load_model('model/fracture_model.h5')

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route pour servir les fichiers uploadés
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    pred_score = None
    image_url = None

    if request.method == 'POST':
        if 'image' not in request.files:
            prediction_text = 'Aucune image détectée.'
        else:
            img_file = request.files['image']
            if img_file.filename == '':
                prediction_text = 'Aucune image sélectionnée.'
            else:
                filename = secure_filename(img_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                img_file.save(filepath)

                # Préparation de l'image
                img = image.load_img(filepath, target_size=(150, 150))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                # Prédiction
                prediction = model.predict(img_array)[0][0]
                pred_score = prediction

                threshold = 0.7
                if prediction >= threshold:
                    prediction_text = 'Fracture détectée.'
                else:
                    prediction_text = 'Pas de fracture détectée.'

                # Créer URL pour afficher l’image uploadée
                image_url = url_for('uploaded_file', filename=filename)

    return render_template('index.html', prediction=prediction_text, pred_score=pred_score, image_url=image_url)


if __name__ == '__main__':
    app.run(debug=True)
