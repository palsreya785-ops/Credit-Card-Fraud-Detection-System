from src.data_preprocessing import load_data, preprocess
from src.train_model import train
from src.predict import evaluate

# Load data
df = load_data()

# Preprocess
X_train, X_test, y_train, y_test = preprocess(df)

# Train
model = train(X_train, y_train)

# Evaluate
evaluate(X_test, y_test)
