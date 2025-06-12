
# 📦 Импорт
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 📥 Загрузка
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")

# 🎯 Целевая переменная
df["Transported"] = df["Transported"].astype(int)

# 🧹 Предобработка
X = df.drop(["Transported", "PassengerId", "Name", "Cabin"], axis=1)
y = df["Transported"]

# Категориальные и числовые признаки
cat_features = X.select_dtypes(include="object").columns.tolist()
num_features = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()

# Преобразователь
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]), num_features),
    
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_features)
])

# 🌲 Модель
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
])

# 📤 Трейн-спліт
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 🏋️‍♀️ Обучение
model.fit(X_train, y_train)

# 🧪 Оценка
y_pred = model.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
print(f"Validation accuracy: {acc:.4f}")

# ✅ Предсказание на тесте
X_test = df_test.drop(["PassengerId", "Name", "Cabin"], axis=1)
predictions = model.predict(X_test)

# 📁 Сохранение сабмісії
submission["Transported"] = predictions.astype(bool)
submission.to_csv("submission.csv", index=False)
