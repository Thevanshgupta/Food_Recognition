from flask import Flask, render_template, request
import os
from food_info import get_food_info
from food_classifier import predict_food  # NEW

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Predict food using ResNet50 classifier
            food_name = predict_food(filepath)
            info = get_food_info(food_name)

            return render_template('result.html', food=food_name, info=info, image=filepath)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
