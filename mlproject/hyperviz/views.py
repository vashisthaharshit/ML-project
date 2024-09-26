import pandas as pd
from django.shortcuts import render
from .utils import train_and_predict, plot_decision_boundary
from sklearn.model_selection import train_test_split

DATASET_PATH = 'C:\\Users\\harshit.vashistha\\django project\\dataset\\placement-dataset.csv'  

def load_data():
    data = pd.read_csv(DATASET_PATH)
    x = data[['cgpa', 'iq']].values  
    y = data['placement'].values  
    return x, y

def index(request):
    plot_image = None
    accuracy = None
    x, y = load_data() 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    if request.method == 'POST':
        model_type = request.POST.get('model')
        params = {}

        if model_type == 'random_forest':
            if request.POST.get('n_estimators') == '':
                params['n_estimators'] = 10
            else:
                params['n_estimators'] = int(request.POST.get('n_estimators'))  

            if request.POST.get('max_depth') == '0':
                params['max_depth'] = None
            else:
                params['max_depth'] = int(request.POST.get('max_depth'))  

            if request.POST.get('min_samples_split') == '':
                params['min_samples_split'] = 2
            else:
                params['min_samples_split'] = int(request.POST.get('min_samples_split'))  

        elif model_type == 'logistic_regression':
            if request.POST.get('C') == '':
                params['C'] = 1.0
            else:
                params['C'] = float(request.POST.get('C'))  

            if request.POST.get('penalty') == '':
                params['penalty'] = 'l2'
            else:
                params['penalty'] = request.POST.get('penalty')

        elif model_type == 'xgboost':
            if request.POST.get('n_estimators') == '':
                params['n_estimators'] = 100
            else:
                params['n_estimators'] = int(request.POST.get('n_estimators'))  

            if request.POST.get('max_depth') == '0':
                params['max_depth'] = None
            else:
                params['max_depth'] = int(request.POST.get('max_depth'))  

            if request.POST.get('learning_rate') == '':
                params['learning_rate'] = 0.1
            else:
                params['learning_rate'] = float(request.POST.get('learning_rate'))  

        model, accuracy = train_and_predict(model_type, params, x_train, y_train, x_test, y_test) 
        plot_image = plot_decision_boundary(x, y, model)


    return render(request, 'index.html', {'plot_image': plot_image, 'accuracy': accuracy})
