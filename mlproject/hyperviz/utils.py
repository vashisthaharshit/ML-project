import matplotlib
matplotlib.use('Agg')  
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.metrics import classification_report, r2_score, mean_absolute_error, mean_squared_error
import base64

def train_and_predict(model_type, params, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
    ])

    if model_type == 'random_forest':
        print(params.get('n_estimators'))
        classifier = RandomForestClassifier(
            n_estimators=params.get('n_estimators'),
            max_depth=params.get('max_depth'),
            min_samples_split=params.get('min_samples_split')
        )
    elif model_type == 'logistic_regression':
        classifier = LogisticRegression(
            C=params.get('C'),
            penalty=params.get('penalty'),
            solver='liblinear'
        )
    elif model_type == 'xgboost':
        classifier = xgb.XGBClassifier(
            n_estimators=params.get('n_estimators'),
            learning_rate=params.get('learning_rate'),
            max_depth=params.get('max_depth')
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline.steps.append(('classifier', classifier))
    pipeline.fit(x_train, y_train)
    
    return pipeline

def train_and_predict_regressor(model_type, params, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
    ])

    if model_type == 'random_forest':
        print(params.get('n_estimators'))
        regressor = RandomForestRegressor(
            n_estimators=params.get('n_estimators'),
            max_depth=params.get('max_depth'),
            min_samples_split=params.get('min_samples_split')
        )
    elif model_type == 'svm':
        regressor = SVR(
            kernel=params.get('kernel'),
            degree=params.get('degree'),
            C=params.get('C')
        )
    elif model_type == 'xgboost':
        regressor = xgb.XGBRegressor(
            n_estimators=params.get('n_estimators'),
            learning_rate=params.get('learning_rate'),
            max_depth=params.get('max_depth')
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline.steps.append(('regressor', regressor))
    pipeline.fit(x_train, y_train)
    r2 = r2_score(y_test, pipeline.predict(x_test))
    mae = mean_absolute_error(y_test, pipeline.predict(x_test))
    mse = mean_squared_error(y_test, pipeline.predict(x_test))
    
    return pipeline, r2, mae, mse, x_test, y_test

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    print(type(model))

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



def plot_regression_surface(X, y, model):
    # Create a figure and axis
    fig, ax = plt.subplots()

    print(type(model))

    # Scatter plot of the original data
    ax.scatter(X[:, 0], y, c='blue', edgecolors='k', marker='o', label='Data points')

    # Generate points for the best fit line
    X_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100).reshape(-1, 1)
    if X.shape[1] > 1:
        X_range = np.column_stack([X_range, np.zeros((100, X.shape[1]-1))])
    y_pred = model.predict(X_range)

    # Plot the best fit line
    ax.plot(X_range[:, 0], y_pred, color='red', linewidth=2, label='Best fit line')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Regression Best Fit Line')
    ax.legend()

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    # Encode the image to base64
    return base64.b64encode(buf.read()).decode('utf-8')
