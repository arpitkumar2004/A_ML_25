# simple placeholder for encoding categorical features
from sklearn.preprocessing import LabelEncoder
import joblib

ENC_PATH = "experiments/models/label_encoder.joblib"

def fit_label_encoder(series):
    le = LabelEncoder()
    vals = series.fillna("NA").astype(str)
    le.fit(vals)
    joblib.dump(le, ENC_PATH)
    return le

def transform_label_encoder(series):
    import joblib
    le = joblib.load(ENC_PATH)
    return le.transform(series.fillna("NA").astype(str))

