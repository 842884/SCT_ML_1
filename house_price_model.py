import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Feature selection
features = ['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF']
X = train_df[features]
y = train_df['SalePrice']
X_test = test_df[features]

# Handle missing values
X = X.fillna(0)
X_test = X_test.fillna(0)

# Model training
model = LinearRegression()
model.fit(X, y)

# Predict
predictions = model.predict(X_test)

# Save submission
submission = sample_submission.copy()
submission['SalePrice'] = predictions
submission.to_csv('submission.csv', index=False)

print("✅ Prediction saved as submission.csv")
