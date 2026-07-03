import streamlit as st
import pandas as pd
from inference import CreditScoreInference


st.set_page_config(
    page_title="Credit Score Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("Credit Score Prediction App")
st.write("Aplikasi ini memprediksi kategori credit score nasabah (Good / Standard / Poor) berdasarkan informasi finansial dan riwayat kredit. Farrel Aziz Sjoekri - 2802566413")

inference = CreditScoreInference()

month_options = [
    "January", "February", "March", "April",
    "May", "June", "July", "August"
]

occupation_options = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer",
    "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
    "Media_Manager", "Musician", "Scientist", "Teacher", "Writer"
]

credit_mix_options = ["Bad", "Standard", "Good"]
payment_min_options = ["Yes", "No"]

payment_behaviour_options = [
    "Low_spent_Small_value_payments",
    "Low_spent_Medium_value_payments",
    "Low_spent_Large_value_payments",
    "High_spent_Small_value_payments",
    "High_spent_Medium_value_payments",
    "High_spent_Large_value_payments"
]

loan_options = inference.loan_types


st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    month = st.selectbox("Month", month_options)
    age = st.number_input("Age", min_value=18, max_value=100, value=24)
    occupation = st.selectbox("Occupation", occupation_options)
    annual_income = st.number_input("Annual Income", min_value=0.0, value=15915.045)
    monthly_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=1256.25375)
    num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, value=8)

with col2:
    num_credit_card = st.number_input("Number of Credit Cards", min_value=0, value=5)
    interest_rate = st.number_input("Interest Rate", min_value=0, value=12)
    num_of_loan = st.number_input("Number of Loans", min_value=0, value=3)
    selected_loans = st.multiselect(
        "Type of Loan",
        loan_options,
        default=["Credit-Builder Loan", "Payday Loan", "Not Specified"]
    )
    delay_from_due_date = st.number_input("Delay from Due Date", min_value=0, value=10)
    num_delayed_payment = st.number_input("Number of Delayed Payments", min_value=0, value=14)

with col3:
    changed_credit_limit = st.number_input("Changed Credit Limit", value=2.71)
    num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, value=4)
    credit_mix = st.selectbox("Credit Mix", credit_mix_options, index=1)
    outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=810.26)
    credit_utilization_ratio = st.number_input("Credit Utilization Ratio", min_value=0.0, value=34.5157)
    credit_history_years = st.number_input("Credit History Years", min_value=0, value=29)
    credit_history_months = st.number_input("Credit History Months", min_value=0, max_value=11, value=10)


st.subheader("Payment Information")

col4, col5, col6 = st.columns(3)

with col4:
    payment_of_min_amount = st.selectbox("Payment of Minimum Amount", payment_min_options, index=1)

with col5:
    total_emi_per_month = st.number_input("Total EMI per Month", min_value=0.0, value=25.4728)
    amount_invested_monthly = st.number_input("Amount Invested Monthly", min_value=0.0, value=75.996)

with col6:
    payment_behaviour = st.selectbox("Payment Behaviour", payment_behaviour_options)
    monthly_balance = st.number_input("Monthly Balance", min_value=0.0, value=314.1566)


credit_history_age = (credit_history_years * 12) + credit_history_months
type_of_loan = ", ".join(selected_loans)


input_data = {
    "Month": month,
    "Age": age,
    "Occupation": occupation,
    "Annual_Income": annual_income,
    "Monthly_Inhand_Salary": monthly_salary,
    "Num_Bank_Accounts": num_bank_accounts,
    "Num_Credit_Card": num_credit_card,
    "Interest_Rate": interest_rate,
    "Num_of_Loan": num_of_loan,
    "Type_of_Loan": type_of_loan,
    "Delay_from_due_date": delay_from_due_date,
    "Num_of_Delayed_Payment": num_delayed_payment,
    "Changed_Credit_Limit": changed_credit_limit,
    "Num_Credit_Inquiries": num_credit_inquiries,
    "Credit_Mix": credit_mix,
    "Outstanding_Debt": outstanding_debt,
    "Credit_Utilization_Ratio": credit_utilization_ratio,
    "Credit_History_Age": credit_history_age,
    "Payment_of_Min_Amount": payment_of_min_amount,
    "Total_EMI_per_month": total_emi_per_month,
    "Amount_invested_monthly": amount_invested_monthly,
    "Payment_Behaviour": payment_behaviour,
    "Monthly_Balance": monthly_balance
}


st.divider()

if st.button("Predict Credit Score"):
    prediction, probabilities = inference.predict(input_data)

    st.subheader("Prediction Result")
    st.success(f"Predicted Credit Score: {prediction}")

    proba_df = pd.DataFrame({
        "Credit Score": list(probabilities.keys()),
        "Probability": list(probabilities.values())
    }).sort_values(by="Probability", ascending=False)

    st.write("Prediction Probability:")
    st.dataframe(proba_df, use_container_width=True)

    st.bar_chart(proba_df.set_index("Credit Score"))