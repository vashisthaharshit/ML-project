import matplotlib
matplotlib.use('Agg')  

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.metrics import classification_report
import base64

def train_and_predict(model_type, params, X_train, y_train, X_test, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
    ])

    if model_type == 'random_forest':
        classifier = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 10),
            max_depth=params.get('max_depth', None),
            min_samples_split=params.get('min_samples_split', 2)
        )
    elif model_type == 'logistic_regression':
        classifier = LogisticRegression(
            C=params.get('C', 1.0),
            penalty=params.get('penalty', 'l2'),
            solver='liblinear'
        )
    elif model_type == 'xgboost':
        classifier = xgb.XGBClassifier(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            max_depth=params.get('max_depth', 3)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline.steps.append(('classifier', classifier))
    pipeline.fit(X_train, y_train)
    accuracy = classification_report(y_test, pipeline.predict(X_test), output_dict=True)
    
    return pipeline, accuracy

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title("Decision Boundary")
    plt.xlabel("CGPA")
    plt.ylabel("IQ")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode('utf-8')
