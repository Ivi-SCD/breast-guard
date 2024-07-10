from flask import Flask, request, render_template
import joblib
import numpy as np

model = joblib.load('./model/breast-guard-model.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            features = [float(request.form[feature]) for feature in [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
            'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]]
            
            features = np.array([features])
            prediction = model.predict(features)
            prediction = 'Malignant' if prediction[0] == 1 else 'Benign'
            
            return render_template('form.html', prediction=prediction)
        except Exception as e:
            return f"Erro: {str(e)}"
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
