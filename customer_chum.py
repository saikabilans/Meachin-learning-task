import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report

df = pd.read_csv("customer_chum.csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())


print("\n===== DATASET SHAPE =====")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== BASIC STATISTICS =====")
print(df.describe())


X = df.drop(columns=["customer_id", "churn"])

y = df["churn"]


categorical_columns = ["gender", "contract_type"]

numerical_columns = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n===== DATA SPLIT =====")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

model.fit(X_train, y_train)

print("\n===== MODEL TRAINING =====")
print("Model trained successfully!")


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\n===== ACCURACY =====")
print(f"Accuracy: {accuracy:.2f}")


precision = precision_score(
    y_test,
    y_pred,
    pos_label="Yes"
)

print("\n===== PRECISION =====")
print(f"Precision: {precision:.2f}")


print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))


cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n===== CROSS VALIDATION =====")
print("Fold scores:", cv_scores)

print(f"Average validation accuracy: {cv_scores.mean():.2f}")

