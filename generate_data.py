import pandas as pd
import numpy as np
import os

# Create data folder if not exists
os.makedirs("data", exist_ok=True)

np.random.seed(42)

n = 5000

# Generate normal transactions (95%)
normal_size = int(n * 0.95)
fraud_size = n - normal_size

# Normal transactions
normal_data = pd.DataFrame({
    "amount": np.random.normal(200, 50, normal_size), # smaller amounts
    "time": np.random.randint(6, 22, normal_size), # daytime
    "location": np.random.randint(1, 50, normal_size),
    "is_fraud": 0
})

# Fraud transactions
fraud_data = pd.DataFrame({
    "amount": np.random.normal(2000, 500, fraud_size), # high amounts
    "time": np.random.randint(0, 6, fraud_size), # odd hours
    "location": np.random.randint(50, 100, fraud_size),
    "is_fraud": 1
})

# Combine both
df = pd.concat([normal_data, fraud_data])

# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# Save file
file_path = os.path.join("data", "synthetic_data.csv")
df.to_csv(file_path, index=False)

print("✅ Synthetic dataset created successfully!")
print(f"📁 Saved at: {file_path}")
print(df.head())
