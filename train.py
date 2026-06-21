import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("Telco-Customer-Churn.csv")

# ---------------- CLEAN DATA ---------------- #
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

# force categorical columns as string (VERY IMPORTANT)
cat_cols = X.select_dtypes(include=["object"]).columns
for col in cat_cols:
    X[col] = X[col].astype(str)

num_cols = X.select_dtypes(exclude=["object"]).columns

# ---------------- PREPROCESSING ---------------- #
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# ---------------- MODEL ---------------- #
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

# ---------------- SPLIT ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- TRAIN ---------------- #
model.fit(X_train, y_train)

# ---------------- SAVE ---------------- #
joblib.dump(model, "churn_pipeline.pkl")

print("MODEL TRAINED SUCCESSFULLY ✔")