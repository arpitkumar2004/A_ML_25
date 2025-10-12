"""IO helpers for models, OOF, and data."""
import joblib
def save_obj(obj, path):
    joblib.dump(obj, path)
def load_obj(path):
    return joblib.load(path)

