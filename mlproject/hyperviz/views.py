import pandas as pd
from django.shortcuts import render
from .utils import train_and_predict, plot_decision_boundary, train_and_predict_regressor, plot_regression_surface
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 

def load_data():
    data = sns.load_dataset('iris')
    x = data[['sepal_length', 'petal_length']].values  
    y = data['species'].values  
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    return x, y

def index(request):
    plot_image = None
    x, y = load_data() 

    if request.method == 'POST':
        model_type = request.POST.get('model')
        params = {}

        if model_type == 'random_forest':
            params['n_estimators'] = int(request.POST.get('n_estimators', 10))
            params['max_depth'] = int(request.POST.get('max_depth', 0)) or None
            params['min_samples_split'] = int(request.POST.get('min_samples_split', 2))

        elif model_type == 'logistic_regression':
            params['C'] = float(request.POST.get('C', 1.0))
            params['penalty'] = request.POST.get('penalty', 'l2')

        elif model_type == 'xgboost':
            params['n_estimators'] = int(request.POST.get('n_estimators', 100))
            params['max_depth'] = int(request.POST.get('max_depth', 0)) or None
            params['learning_rate'] = float(request.POST.get('learning_rate', 0.1))

        model = train_and_predict(model_type, params, x, y) 
        print('my', type(model))
        plot_image = plot_decision_boundary(x, y, model)

    return render(request, 'index.html', {'plot_image': plot_image})

def load_regressor():
    data = sns.load_dataset('tips')
    x = data[['total_bill', 'size']].values  
    y = data['tip'].values  
    return x, y

def regressor(request):
    mae = None
    mse = None
    r2 = None
    plot_image = None  # Initialize plot image
    x, y = load_regressor()

    if request.method == 'POST':
        model_type = request.POST.get('model')
        params = {}

        if model_type == 'random_forest':
            params['n_estimators'] = int(request.POST.get('n_estimators', 10))
            params['max_depth'] = int(request.POST.get('max_depth', 0)) or None
            params['min_samples_split'] = int(request.POST.get('min_samples_split', 2))

        elif model_type == 'svm':  # Ensure case matches the form
            params['C'] = float(request.POST.get('C', 1.0))
            params['degree'] = int(request.POST.get('degree', 1))
            params['kernel'] = request.POST.get('kernel', 'linear')

        elif model_type == 'xgboost':
            params['n_estimators'] = int(request.POST.get('n_estimators', 100))
            params['max_depth'] = int(request.POST.get('max_depth', 0)) or None
            params['learning_rate'] = float(request.POST.get('learning_rate', 0.1))

        model, r2, mae, mse, x_test, y_test = train_and_predict_regressor(model_type, params, x, y)
        plot_image = plot_regression_surface(x_test, y_test, model)

    return render(request, 'regressor.html', {'r2': r2, 'mae': mae, 'mse': mse, 'plot_image': plot_image})
