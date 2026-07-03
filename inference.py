import joblib
import pandas as pd


class CreditScoreInference:
 

    def __init__(self, model_dir: str = "model"):
        self.model = joblib.load(f"{model_dir}/best_credit_score_model.pkl")
        self.feature_columns = joblib.load(f"{model_dir}/feature_columns.pkl")
        self.loan_types = joblib.load(f"{model_dir}/types_of_loans.pkl")

    def prepare_input(self, input_data: dict) -> pd.DataFrame:
        """
        Mengubah satu input mentah (dict) menjadi DataFrame satu baris yang
        sudah memiliki fitur binary Loan_* (hasil feature engineering
        Type_of_Loan) dan kolom sesuai urutan feature_columns hasil training.
        """
        input_df = pd.DataFrame([input_data])

        if "Type_of_Loan" in input_df.columns:
            loan_text = str(input_df.loc[0, "Type_of_Loan"])

            for loan in self.loan_types:
                col_name = "Loan_" + loan.replace(" ", "_").replace("-", "_")
                input_df[col_name] = int(loan in loan_text)

            input_df = input_df.drop(columns=["Type_of_Loan"])

        # Kolom fitur yang tidak ada pada input (mis. jenis pinjaman yang
        # tidak dipilih) diisi 0 agar bentuk data konsisten dengan saat training.
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[self.feature_columns]

        return input_df

    def predict(self, input_data: dict):
        input_df = self.prepare_input(input_data)

        prediction = self.model.predict(input_df)[0]
        probabilities = self.model.predict_proba(input_df)[0]

        proba_result = {
            class_name: float(prob)
            for class_name, prob in zip(self.model.classes_, probabilities)
        }

        return prediction, proba_result


if __name__ == "__main__":
    sample_input = {
        "Month": "March",
        "Age": 24,
        "Occupation": "Architect",
        "Annual_Income": 15915.045,
        "Monthly_Inhand_Salary": 1256.25375,
        "Num_Bank_Accounts": 8,
        "Num_Credit_Card": 5,
        "Interest_Rate": 12,
        "Num_of_Loan": 3,
        "Type_of_Loan": "Credit-Builder Loan, Payday Loan, and Not Specified",
        "Delay_from_due_date": 10,
        "Num_of_Delayed_Payment": 14,
        "Changed_Credit_Limit": 2.71,
        "Num_Credit_Inquiries": 4,
        "Credit_Mix": "Standard",
        "Outstanding_Debt": 810.26,
        "Credit_Utilization_Ratio": 34.515710633531164,
        "Credit_History_Age": 358,
        "Payment_of_Min_Amount": "No",
        "Total_EMI_per_month": 25.47281887542715,
        "Amount_invested_monthly": 75.9959913400093,
        "Payment_Behaviour": "Low_spent_Small_value_payments",
        "Monthly_Balance": 314.1565647845636,
    }

    inference = CreditScoreInference()
    prediction, probabilities = inference.predict(sample_input)

    print("Prediction:", prediction)
    print("Probabilities:", probabilities)