import joblib
import os

# Load the model
model_path = os.path.join("model", "final_model.pkl")
model = joblib.load(model_path)

# Print info
try:
    print("✅ Number of input features:", model.n_features_in_)
except:
    print("❌ model.n_features_in_ not available")

try:
    print("✅ Feature names:")
    print(model.feature_names_in_)
except:
    print("❌ model.feature_names_in_ not available")
