"""
train_pipeline.py
==================
Struktur OOP:
- DataCleaner      -> membersihkan dirty/invalid value pada raw data
- FeatureEngineer   -> feature engineering kolom Type_of_Loan menjadi fitur binary
- Preprocessor      -> split X/y, train-test split, dan ColumnTransformer
- ModelTrainer      -> melatih setiap kandidat model di atas preprocessing pipeline
- ModelEvaluator     -> menghitung metrik evaluasi & classification report
- ModelPipeline      -> orchestrator: menjalankan seluruh alur end-to-end + MLflow

Cara pakai:
    python train_pipeline.py [path_ke_data.csv]

Jika argumen path tidak diberikan, pipeline akan mencari file data_A.csv
di direktori yang sama (sesuaikan dengan nama file dataset Anda, misalnya
data_A (2).csv).
"""

import os
import re
import sys
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# 1. DATA CLEANING
# ---------------------------------------------------------------------------
class DataCleaner:
    """
    Bertanggung jawab membersihkan raw data sebelum masuk ke tahap
    preprocessing/modelling:
      - drop kolom identifier (tidak informatif untuk model)
      - normalisasi dirty value pada kolom kategorikal (mis. tanda garis
        bawah berulang, atau simbol acak seperti !@9#%8)
      - konversi kolom numerik yang masih bertipe object (mengandung
        karakter underscore)
      - konversi Credit_History_Age (format teks X Years Y Months) menjadi bulan
      - menandai nilai yang secara logis tidak valid (mis. umur negatif)
        sebagai missing value (NaN) agar ditangani lewat imputasi
    """

    # Kolom numerik yang pada dataset asli masih bertipe object karena
    # mengandung karakter underscore. Monthly_Balance ikut dimasukkan
    # karena pada data_A kolom ini juga ditemukan mengandung underscore,
    # tidak hanya Age, Annual_Income, dsb.
    NUMERIC_OBJECT_COLS = [
        "Age",
        "Annual_Income",
        "Num_of_Loan",
        "Num_of_Delayed_Payment",
        "Changed_Credit_Limit",
        "Outstanding_Debt",
        "Amount_invested_monthly",
        "Monthly_Balance",
    ]

    DROP_COLS = ["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN"]

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        # raw_data disimpan terpisah karena beberapa dirty-value check
        # (mis. double underscore) perlu dicek dari nilai string asli,
        # bukan dari hasil strip underscore yang sudah dilakukan di awal.
        self.raw_data = data.copy()

    def clean_numeric_column(self, col: str):
        self.data[col] = self.data[col].astype(str).str.replace("_", "", regex=False)
        self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

    def convert_credit_history_age(self, value):
        if pd.isnull(value):
            return np.nan

        value = str(value)

        years = re.search(r"(\d+)\sYears?", value)
        months = re.search(r"(\d+)\sMonths?", value)

        year_value = int(years.group(1)) if years else 0
        month_value = int(months.group(1)) if months else 0

        return (year_value * 12) + month_value

    def clean(self) -> pd.DataFrame:
        self.data = self.data.drop(columns=self.DROP_COLS)

        self.data["Occupation"] = self.data["Occupation"].replace("_______", np.nan)
        self.data["Credit_Mix"] = self.data["Credit_Mix"].replace("_", np.nan)
        self.data["Payment_Behaviour"] = self.data["Payment_Behaviour"].replace("!@9#%8", np.nan)
        self.data["Payment_of_Min_Amount"] = self.data["Payment_of_Min_Amount"].replace("NM", np.nan)

        for col in self.NUMERIC_OBJECT_COLS:
            self.clean_numeric_column(col)

        self.data["Credit_History_Age"] = self.data["Credit_History_Age"].apply(
            self.convert_credit_history_age
        )

        self.handle_invalid_values()

        return self.data

    def handle_invalid_values(self):
        self.data.loc[(self.data["Age"] < 0) | (self.data["Age"] > 100), "Age"] = np.nan
        self.data.loc[
            (self.data["Num_Bank_Accounts"] < 0) | (self.data["Num_Bank_Accounts"] > 20),
            "Num_Bank_Accounts",
        ] = np.nan
        self.data.loc[self.data["Num_Credit_Card"] > 20, "Num_Credit_Card"] = np.nan
        self.data.loc[self.data["Interest_Rate"] > 100, "Interest_Rate"] = np.nan

        self.data.loc[
            (self.data["Num_of_Loan"] < 0) | (self.data["Num_of_Loan"] > 20), "Num_of_Loan"
        ] = np.nan
        # Delay negatif dianggap sebagai bayar lebih awal -> 0, bukan missing value
        self.data.loc[self.data["Delay_from_due_date"] < 0, "Delay_from_due_date"] = 0

        self.data.loc[
            (self.data["Num_of_Delayed_Payment"] < 0) | (self.data["Num_of_Delayed_Payment"] > 100),
            "Num_of_Delayed_Payment",
        ] = np.nan

        self.data.loc[self.data["Num_Credit_Inquiries"] > 100, "Num_Credit_Inquiries"] = np.nan

        # Dirty sentinel value dengan pola double underscore, mis.
        # "__10000__" atau "__-333333333333333333333333333__". Jika hanya
        # di-strip underscore-nya akan menghasilkan angka yang tidak masuk
        # akal, sehingga baris ini ditandai sebagai missing value.
        dirty_amount_mask = self.raw_data["Amount_invested_monthly"].astype(str).str.contains(
            "__", regex=False
        )
        self.data.loc[dirty_amount_mask, "Amount_invested_monthly"] = np.nan

        dirty_balance_mask = self.raw_data["Monthly_Balance"].astype(str).str.contains(
            "__", regex=False
        )
        self.data.loc[dirty_balance_mask, "Monthly_Balance"] = np.nan


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
class FeatureEngineer:
    """
    Mengubah kolom `Type_of_Loan` (satu nasabah dapat memiliki beberapa jenis
    pinjaman sekaligus dalam satu string) menjadi beberapa fitur binary,
    satu kolom per jenis pinjaman.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.loan_types = [
            "Auto Loan",
            "Credit-Builder Loan",
            "Debt Consolidation Loan",
            "Home Equity Loan",
            "Mortgage Loan",
            "Payday Loan",
            "Personal Loan",
            "Student Loan",
            "Not Specified",
        ]

    def create_loan_features(self):
        for loan in self.loan_types:
            col_name = "Loan_" + loan.replace(" ", "_").replace("-", "_")
            self.data[col_name] = (
                self.data["Type_of_Loan"].fillna("").str.contains(loan, regex=False).astype(int)
            )

        self.data = self.data.drop(columns=["Type_of_Loan"])

        return self.data, self.loan_types


# ---------------------------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------------------------
class Preprocessor:
    """
    Bertanggung jawab menyiapkan data sebelum modelling:
      - memisahkan fitur (X) dan target (y)
      - train/test split (stratified, menjaga proporsi kelas)
      - identifikasi fitur numerik vs kategorikal
      - membangun ColumnTransformer:
          * numerik  -> median imputation + StandardScaler
          * kategorik -> most-frequent imputation + OrdinalEncoder

    Catatan: OrdinalEncoder dipakai (bukan OneHotEncoder) mengikuti notebook
    referensi, karena jumlah kategori pada beberapa kolom (mis. Occupation)
    cukup banyak dan tree-based model (Random Forest/Extra Trees, model
    terbaik pada eksperimen ini) dapat bekerja baik dengan encoding ordinal
    tanpa menambah dimensi fitur secara signifikan.
    """

    def __init__(self, target_col: str = "Credit_Score", test_size: float = 0.2, random_state: int = 42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

    def split_features_target(self, data: pd.DataFrame):
        X = data.drop(columns=[self.target_col])
        y = data[self.target_col]
        return X, y

    def identify_feature_types(self, X: pd.DataFrame):
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include="object").columns.tolist()
        return numeric_features, categorical_features

    def split_train_test(self, X: pd.DataFrame, y: pd.Series):
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

    def build_transformer(self, numeric_features, categorical_features) -> ColumnTransformer:
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])

        return ColumnTransformer(transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ])


# ---------------------------------------------------------------------------
# 4. TRAINING
# ---------------------------------------------------------------------------
class ModelTrainer:
    """
    Bertanggung jawab melatih setiap kandidat model machine learning di atas
    preprocessing pipeline yang sama (mencegah data leakage, memastikan
    konsistensi transformasi antar model).

    Daftar kandidat model merangkum kedua tahap eksperimen pada notebook:
    baseline (Logistic Regression, Decision Tree, Random Forest) dan final
    model experiment (variasi Random Forest & Extra Trees, Gradient
    Boosting, Hist Gradient Boosting).
    """

    def __init__(self, preprocessor: ColumnTransformer):
        self.preprocessor = preprocessor

    def get_candidate_models(self) -> dict:
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=42, class_weight="balanced"
            ),
            "Random Forest Baseline": RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight="balanced"
            ),
            "Random Forest 150": RandomForestClassifier(
                n_estimators=150, random_state=42, class_weight="balanced", n_jobs=-1
            ),
            "Random Forest No Class Weight": RandomForestClassifier(
                n_estimators=150, random_state=42, n_jobs=-1
            ),
            "Extra Trees 150": ExtraTreesClassifier(
                n_estimators=150, max_depth=20, min_samples_leaf=5, random_state=42, class_weight="balanced", n_jobs=-1
            ),
            "Extra Trees No Class Weight": ExtraTreesClassifier(
                n_estimators=150, random_state=42, n_jobs=-1
            ),
            # Varian dengan 100 tree: pada eksperimen di dataset ini terbukti
            # F1-nya setara/lebih baik dari versi 150 tree, dengan ukuran file
            # model jauh lebih kecil (penting untuk deployment/GitHub).
            "Extra Trees 100 No Class Weight": ExtraTreesClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
        }

    def train(self, model, X_train, y_train) -> Pipeline:
        pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        return pipeline


# ---------------------------------------------------------------------------
# 5. EVALUATION
# ---------------------------------------------------------------------------
class ModelEvaluator:
    """
    Bertanggung jawab menghitung metrik evaluasi (accuracy, precision,
    recall, weighted F1-score), classification report, dan confusion matrix
    untuk model yang sudah dilatih.

    Weighted F1-score dipakai sebagai kriteria utama pemilihan model terbaik
    karena target (Credit_Score) mengalami class imbalance (Standard 53%,
    Poor 29%, Good 17%), sehingga accuracy saja berpotensi bias terhadap
    kelas mayoritas.
    """

    def evaluate(self, pipeline: Pipeline, X_test, y_test) -> dict:
        y_pred = pipeline.predict(X_test)

        return {
            "y_pred": y_pred,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
        }

    def classification_report_text(self, y_test, y_pred) -> str:
        return classification_report(y_test, y_pred)

    def confusion_matrix_df(self, y_test, y_pred, labels) -> pd.DataFrame:
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        return pd.DataFrame(cm, index=labels, columns=labels)


# ---------------------------------------------------------------------------
# 6. ORCHESTRATOR
# ---------------------------------------------------------------------------
class ModelPipeline:
    """
    Orchestrator yang menjalankan seluruh alur end-to-end:
    load data -> cleaning -> feature engineering -> preprocessing ->
    training & evaluasi setiap model (dicatat ke MLflow) -> pemilihan model
    terbaik -> simpan model + artifact pendukung + test case untuk
    deployment testing.
    """

    def __init__(self, data_path: str, experiment_name: str = "Credit_Score_Model_Training"):
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.model_folder = "model"
        self.test_case_folder = "test_cases_file"
        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(self.test_case_folder, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path)

    def save_outputs(self, best_pipeline, feature_columns, loan_types, results_df):
        # compress=3 dipakai agar ukuran file model tetap ramah untuk di-commit
        # ke GitHub (batas GitHub adalah 100MB per file). Untuk model tree
        # ensemble seperti Random Forest/Extra Trees, compress=3 biasanya
        # memangkas ukuran file sekitar 5-6x tanpa mengubah hasil prediksi
        # sedikit pun (joblib.load akan otomatis mendekompresnya).
        joblib.dump(
            best_pipeline,
            os.path.join(self.model_folder, "best_credit_score_model.pkl"),
            compress=9,
        )
        joblib.dump(feature_columns, os.path.join(self.model_folder, "feature_columns.pkl"))
        joblib.dump(loan_types, os.path.join(self.model_folder, "types_of_loans.pkl"))
        results_df.to_csv(os.path.join(self.model_folder, "model_comparison_results.csv"), index=False)

    def save_test_cases(self, X_test, y_test, best_pipeline) -> pd.DataFrame:
        """
        Menyimpan satu test case per kelas target (Good, Poor, Standard)
        beserta label aktual & prediksinya, untuk dipakai sebagai test case
        representatif ketika menguji hasil deployment (lihat test_inference.py).
        """
        test_data = X_test.copy()
        test_data["Actual_Credit_Score"] = y_test.values
        test_data["Predicted_Credit_Score"] = best_pipeline.predict(X_test)

        test_cases = []
        for label in ["Good", "Poor", "Standard"]:
            sample = test_data[test_data["Actual_Credit_Score"] == label].head(1)
            test_cases.append(sample)

        test_cases_df = pd.concat(test_cases)

        output_path = os.path.join(self.test_case_folder, "credit_score_testing.csv")
        test_cases_df.to_csv(output_path, index=False)

        print(f"\nTest case (1 sampel per kelas) disimpan di: {output_path}")
        return test_cases_df

    def run(self):
        # 1) Load & clean data --------------------------------------------------
        data = self.load_data()

        cleaner = DataCleaner(data)
        cleaned_data = cleaner.clean()

        # 2) Feature engineering --------------------------------------------------
        engineer = FeatureEngineer(cleaned_data)
        model_data, loan_types = engineer.create_loan_features()

        # 3) Preprocessing setup --------------------------------------------------
        prep = Preprocessor(target_col="Credit_Score", test_size=0.2, random_state=42)
        X, y = prep.split_features_target(model_data)
        numeric_features, categorical_features = prep.identify_feature_types(X)
        X_train, X_test, y_train, y_test = prep.split_train_test(X, y)
        transformer = prep.build_transformer(numeric_features, categorical_features)

        print(f"Jumlah fitur numerik   : {len(numeric_features)}")
        print(f"Jumlah fitur kategorikal: {len(categorical_features)}")
        print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")

        # 4) & 5) Training + evaluasi setiap kandidat model, dicatat ke MLflow ----
        trainer = ModelTrainer(transformer)
        evaluator = ModelEvaluator()

        mlflow.set_experiment(self.experiment_name)

        results = []
        best_pipeline = None
        best_model_name = None
        best_f1 = -1.0

        for model_name, model in trainer.get_candidate_models().items():
            with mlflow.start_run(run_name=model_name):
                pipeline = trainer.train(model, X_train, y_train)
                metrics = evaluator.evaluate(pipeline, X_test, y_test)
                y_pred = metrics.pop("y_pred")

                mlflow.log_param("model_name", model_name)
                if hasattr(model, "n_estimators"):
                    mlflow.log_param("n_estimators", model.n_estimators)
                if hasattr(model, "class_weight"):
                    mlflow.log_param("class_weight", str(model.class_weight))

                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                report = evaluator.classification_report_text(y_test, y_pred)
                report_path = f"classification_report_{model_name.replace(' ', '_')}.txt"
                with open(report_path, "w") as f:
                    f.write(report)
                mlflow.log_artifact(report_path)
                os.remove(report_path)

                # serialization_format="cloudpickle" dipakai agar seluruh isi
                # Pipeline (ColumnTransformer + estimator) dapat di-load ulang
                # dengan aman lintas versi scikit-learn/MLflow.
                mlflow.sklearn.log_model(pipeline, "model", serialization_format="cloudpickle")


                results.append({
                    "Model": model_name,
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1 Score": metrics["f1_score"],
                })

                print("=" * 60)
                print(model_name)
                print("Accuracy :", metrics["accuracy"])
                print("Precision:", metrics["precision"])
                print("Recall   :", metrics["recall"])
                print("F1 Score :", metrics["f1_score"])

                if metrics["f1_score"] > best_f1:
                    best_f1 = metrics["f1_score"]
                    best_pipeline = pipeline
                    best_model_name = model_name

        results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)

        # Catat run tambahan yang menandai model juara, agar mudah ditelusuri di MLflow UI
        with mlflow.start_run(run_name=f"BEST_MODEL_{best_model_name}"):
            mlflow.set_tag("stage", "champion")
            mlflow.log_param("best_model_name", best_model_name)
            mlflow.log_metric("best_f1_score", best_f1)
            mlflow.sklearn.log_model(best_pipeline, "best_model", serialization_format="cloudpickle")

        # 6) Simpan model & artifact pendukung untuk deployment -------------------
        self.save_outputs(best_pipeline, X.columns.tolist(), loan_types, results_df)
        self.save_test_cases(X_test, y_test, best_pipeline)

        print("\nBest Model:", best_model_name)
        print("Best F1 Score:", best_f1)
        print("\nModel comparison results:")
        print(results_df.to_string(index=False))

        return best_pipeline, best_model_name, best_f1, results_df


if __name__ == "__main__":
    # Path data bisa diberikan lewat argumen command line, contoh:
    #   python train_pipeline.py "data_A (2).csv"
    default_path = "data_A (2).csv"
    data_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    pipeline = ModelPipeline(data_path)
    pipeline.run()