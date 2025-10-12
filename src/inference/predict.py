import joblib
def load_model(path):
    return joblib.load(path)

def predict_with_model(model, X):
    return model.predict(X)
