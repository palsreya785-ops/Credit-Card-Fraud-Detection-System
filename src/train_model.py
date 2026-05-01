from sklearn.ensemble import RandomForestClassifier
import joblib

def train(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")
    return model
