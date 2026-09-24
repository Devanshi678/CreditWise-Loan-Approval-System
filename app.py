import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CreditWise - Loan Approval System",
    page_icon="🏦",
    layout="centered",
)

st.title("🏦 CreditWise Loan Approval System")
st.write(
    "Fill in the applicant's financial and personal details below to predict loan approval status."
)


@st.cache_resource
def load_pipeline():
    return joblib.load("loan_model.pkl")


model = load_pipeline()

st.subheader("Applicant Details")

col1, col2 = st.columns(2)
with col1:
    applicant_income = st.number_input(
        "Applicant Income (₹)", min_value=0.0, value=50000.0, step=1000.0
    )
    coapplicant_income = st.number_input(
        "Coapplicant Income (₹)", min_value=0.0, value=15000.0, step=1000.0
    )
    age = st.number_input("Age", min_value=18, max_value=100, value=32)
    dependents = st.number_input(
        "Dependents", min_value=0, max_value=10, value=1
    )
    credit_score = st.number_input(
        "Credit Score", min_value=300, max_value=900, value=710
    )

with col2:
    loan_amount = st.number_input(
        "Loan Amount (₹)", min_value=0.0, value=200000.0, step=5000.0
    )
    loan_term = st.number_input(
        "Loan Term (Months)", min_value=12, max_value=360, value=48, step=12
    )
    existing_loans = st.number_input(
        "Existing Loans", min_value=0, max_value=10, value=1
    )
    savings = st.number_input(
        "Savings (₹)", min_value=0.0, value=120000.0, step=5000.0
    )
    collateral_value = st.number_input(
        "Collateral Value (₹)", min_value=0.0, value=300000.0, step=5000.0
    )

# Automatic DTI Ratio Calculation
total_income = applicant_income + coapplicant_income
monthly_installment = loan_amount / loan_term if loan_term > 0 else 0.0
dti_ratio = (
    round(monthly_installment / total_income, 4) if total_income > 0 else 0.0
)

st.markdown("---")
st.text_input(
    "Calculated DTI Ratio (Auto-generated)",
    value=f"{dti_ratio:.4f} ({dti_ratio * 100:.2f}%)",
    disabled=True,
    help="DTI is calculated as: (Monthly Loan Installment) / (Total Income)",
)

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    education_level = st.selectbox(
        "Education Level",
        ["Graduate", "Not Graduate", "Undergraduate", "Postgraduate"],
    )
    employment_status = st.selectbox(
        "Employment Status",
        ["Salaried", "Self-employed", "Business", "Contract", "Unemployed"],
    )
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    gender = st.selectbox("Gender", ["Male", "Female"])

with col4:
    loan_purpose = st.selectbox(
        "Loan Purpose", ["Personal", "Car", "Business", "Home", "Education"]
    )
    property_area = st.selectbox(
        "Property Area", ["Urban", "Semiurban", "Rural"]
    )
    employer_category = st.selectbox(
        "Employer Category",
        [
            "Private",
            "Government",
            "MNC",
            "Business",
            "Self",
            "Unemployed",
        ],
    )

st.markdown("---")

if st.button("Evaluate Application", type="primary", use_container_width=True):
    input_data = pd.DataFrame(
        [
            {
                "Applicant_Income": applicant_income,
                "Coapplicant_Income": coapplicant_income,
                "Employment_Status": employment_status,
                "Age": age,
                "Marital_Status": marital_status,
                "Dependents": dependents,
                "Credit_Score": credit_score,
                "Existing_Loans": existing_loans,
                "DTI_Ratio": dti_ratio,
                "Savings": savings,
                "Collateral_Value": collateral_value,
                "Loan_Amount": loan_amount,
                "Loan_Term": loan_term,
                "Loan_Purpose": loan_purpose,
                "Property_Area": property_area,
                "Education_Level": education_level,
                "Gender": gender,
                "Employer_Category": employer_category,
            }
        ]
    )

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("🎉 **Loan Application Status: APPROVED**")
    else:
        st.error("❌ **Loan Application Status: REJECTED**")