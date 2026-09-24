# 🏦 CreditWise — Loan Approval System

An end-to-end machine learning pipeline that predicts whether a loan application should be **Approved** or **Rejected**, built for **SecureTrust Bank** and deployed as an interactive Streamlit web app.

---

## 📌 Problem Statement

SecureTrust Bank offers personal and home loans to customers across urban and rural regions of India. Every day, hundreds of customers apply for loans through online and branch channels. Until now, applications were reviewed **manually** — loan officers checked income proofs, employment details, credit history, and other documents by hand. This process was slow, subjective, and inconsistent, exposing the bank to two major risks:

1. **Financial Loss** — high-risk applicants mistakenly get approved, driving up loan defaults.
2. **Revenue Loss** — creditworthy applicants get wrongfully rejected, causing customer churn.

**Goal:** Design a machine learning system that learns from historical loan decisions and produces fast, consistent, and unbiased Approve/Reject recommendations before final human sign-off.

### Business Optimization Hierarchy

| Priority | Objective | Why |
|---|---|---|
| **1 — Primary** | Minimize **False Positives** | Avoid approving high-risk borrowers → prevents direct financial defaults |
| **2 — Secondary** | Minimize **False Negatives** | Avoid rejecting creditworthy borrowers → protects loan origination revenue |

---

## 📊 Dataset

`loan_approval_data.csv` — **1,000 historical applicant records** across **20 features** covering personal, employment, debt, collateral, and credit-bureau data.

| Column | Description |
|---|---|
| Applicant_ID | Unique applicant ID |
| Applicant_Income | Monthly income of applicant |
| Coapplicant_Income | Monthly income of co-applicant |
| Employment_Status | Salaried / Self-employed / Business / Contract / Unemployed |
| Age | Applicant age |
| Marital_Status | Married / Single |
| Dependents | Number of dependents |
| Credit_Score | Credit bureau score |
| Existing_Loans | Number of already-running loans |
| DTI_Ratio | Debt-to-Income ratio |
| Savings | Savings balance |
| Collateral_Value | Estimated cash value of pledged collateral |
| Loan_Amount | Loan amount requested |
| Loan_Term | Loan duration (months) |
| Loan_Purpose | Home / Education / Personal / Business / Car |
| Property_Area | Urban / Semi-Urban / Rural |
| Education_Level | Graduate / Postgraduate / Undergraduate / Not Graduate |
| Gender | Male / Female |
| Employer_Category | Government / Private / MNC / Self / Business / Unemployed |
| **Loan_Approved** *(target)* | `Yes` = Approved, `No` = Rejected |

**Class balance:** 70.2% Rejected vs. 29.8% Approved — a meaningful imbalance factored into model evaluation.

---

## 🧹 Data Preprocessing Pipeline

1. **Drop identifier** — `Applicant_ID` removed (no predictive signal).
2. **Handle missing values** — ~50 missing values per column, imputed with `SimpleImputer`:
   - Numeric columns → mean
   - Categorical columns → mode (`most_frequent`)
3. **Encode categorical features**
   - `OrdinalEncoder` for `Education_Level`
   - `OneHotEncoder` for nominal fields: `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`
4. **Scale numeric features** — `StandardScaler` applied to all continuous variables to prevent magnitude bias in distance-based and gradient-based models.
5. **Encode target** — `Loan_Approved`: `Yes → 1`, `No → 0`.

All steps are wrapped in a single scikit-learn `Pipeline` + `ColumnTransformer` (see [`final_model.py`](#-project-structure)) so the exact same transformations are applied at inference time.

### Key EDA Findings
- **Credit_Score** — strongest **positive** predictor of approval.
- **DTI_Ratio** — strongest **negative** predictor; higher debt-to-income drives rejection.
- No destructive outliers found across income, credit score, DTI, savings, age, or loan amount.

---

## 🤖 Model Training & Evaluation

Data was split 80/20 (stratified) → 200 test samples (139 actual rejections, 61 actual approvals). Three classifiers were trained and compared:

| Model | Precision | Recall | Accuracy | FP | FN |
|---|---|---|---|---|---|
| Logistic Regression | 78.33% | 77.05% | 86.50% | 13 | 14 |
| KNN (Default) | 62.75% | 52.46% | 76.00% | 19 | 29 |
| KNN (Tuned, GridSearchCV, k=13) | 73.17% | 49.18% | 79.00% | 11 | 31 |
| **Gaussian Naive Bayes** | **80.36%** | **73.77%** | **86.50%** | **11** | **16** |

### ✅ Final Model: Gaussian Naive Bayes

Selected as the production model because it:
- Achieves the **highest precision (80.36%)**, tying for the lowest false-positive count → minimizes default risk (primary business goal).
- Maintains a **strong recall (73.77%)**, far outperforming tuned KNN → protects loan origination revenue (secondary business goal).
- Delivers the **lowest combined error** (FP + FN = 27) alongside top-tier accuracy.

> **Key learning:** Accuracy alone is misleading on imbalanced data — precision/recall trade-offs and confusion-matrix costs matter far more for financial risk decisions.

---

## 🚀 Deployment

The trained pipeline is serialized with `joblib` (`loan_model.pkl`) and served through a **Streamlit** web app (`app.py`) that lets a loan officer:

- Enter applicant financial & personal details (income, credit score, savings, loan amount, etc.)
- Auto-calculates **DTI Ratio** from income and loan installment
- Get an instant **APPROVED ✅ / REJECTED ❌** decision badge

This cuts application review time from **hours/days to seconds**, with consistent, unbiased decisioning across every branch.

---

## 📁 Project Structure

```
CreditWise-Loan-Approval-System/
├── app.py                   # Streamlit web application
├── final_model.py           # Data pipeline + Gaussian Naive Bayes training script
├── loan_approval_data.csv   # Historical applicant dataset (1,000 records)
├── loan_model.pkl           # Serialized trained model pipeline
├── mini_project_1.ipynb     # Exploratory notebook (EDA, preprocessing, model comparison)
├── requirement.txt          # Python dependencies
└── README.md                # Project documentation (this file)
```

---

## ⚙️ Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/CreditWise-Loan-Approval-System.git
cd CreditWise-Loan-Approval-System
```

### 2. Install dependencies
```bash
pip install -r requirement.txt
```

### 3. Train the model (optional — `loan_model.pkl` is already included)
```bash
python final_model.py
```
This reads `loan_approval_data.csv`, fits the preprocessing + Gaussian Naive Bayes pipeline, and saves `loan_model.pkl`.

### 4. Run the app
```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (usually `http://localhost:8501`) and fill in applicant details to get an instant loan decision.

---

## 🛠️ Tech Stack

- **Python 3**
- **pandas / numpy** — data handling
- **scikit-learn** — preprocessing pipelines, model training & evaluation
- **matplotlib / seaborn** — exploratory data analysis & visualization
- **Streamlit** — interactive web deployment
- **joblib** — model serialization

---

## 📈 Future Improvements

- Address class imbalance with techniques like SMOTE or class-weighting.
- Add SHAP/feature-importance explainability to the app for loan-officer transparency.
- Track model performance drift with periodic re-training on new applications.
- Add authentication and audit logging for production banking use.

---

## 👤 Author

Built as a machine learning case study for **SecureTrust Bank's CreditWise Loan Approval System**.
