# Demo prediction for spam classifier
import joblib
from sklearn.datasets import fetch_openml
import pandas as pd

def main():
    clf = joblib.load('model.joblib')
    data = fetch_openml('spambase', version=1, as_frame=True)
    X = data.data
    sample = X.iloc[:5]
    preds = clf.predict(sample)
    print('Predictions for first 5 samples:', preds)

if __name__ == '__main__':
    main()
