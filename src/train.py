# Train a spam classifier using the Spambase dataset from OpenML
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

def main():
    print('Downloading spambase dataset from OpenML (may take a minute)...')
    data = fetch_openml('spambase', version=1, as_frame=True)
    X = data.data
    y = data.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(clf, 'model.joblib')
    print('Saved model to model.joblib')

if __name__ == '__main__':
    main()
