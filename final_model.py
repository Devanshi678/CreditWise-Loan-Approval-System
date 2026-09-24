import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# -------------------- Load Data --------------------
data = pd.read_csv("loan_approval_data.csv")

# Clean whitespace from all string columns
for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].astype(str).str.strip()

# Drop rows with missing target values
data = data.dropna(subset=["Loan_Approved"]).copy()

# Drop ID column if present
if "Applicant_ID" in data.columns:
    data.drop(columns=["Applicant_ID"], inplace=True)

# -------------------- Split Target & Features --------------------
X = data.drop(columns=["Loan_Approved"])
y = data["Loan_Approved"].map({"Yes": 1, "No": 0})

# Keep only rows where target was mapped successfully
valid_idx = y.notna()
X = X[valid_idx]
y = y[valid_idx].astype(int)

# -------------------- Feature Categorization --------------------
ordinal_data_col = ["Education_Level"]
ohe_cols = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category",
]

# Select all numeric columns (float64 and int64)
numerical_col = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

# -------------------- Pipelines --------------------
num_pipeline = Pipeline(
    [
        ("numerical_imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

ordinal_cols_pipeline = Pipeline(
    [
        ("cat_imputer", SimpleImputer(strategy="most_frequent")),
        (
            "oe",
            OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ),
        ),
    ]
)

ohe_cols_pipeline = Pipeline(
    [
        ("cat_imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ohe",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("numerical", num_pipeline, numerical_col),
        ("ordinal_data", ordinal_cols_pipeline, ordinal_data_col),
        ("ohe_data", ohe_cols_pipeline, ohe_cols),
    ]
)

model_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("naive_bayes_model_classifier", GaussianNB()),
    ]
)

# Fit and Save
model_pipeline.fit(X, y)
joblib.dump(model_pipeline, "loan_model.pkl")
print("Model pipeline trained and saved successfully to loan_model.pkl!")
