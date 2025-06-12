
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

# 📥 Загрузка данных
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")

# 🎯 Целевая переменная
df["Transported"] = df["Transported"].astype(int)

# 🧠 Feature Engineering
def enrich(df):
    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    df["Group"] = df["PassengerId"].apply(lambda x: x.split("_")[0])
    df["NameLength"] = df["Name"].fillna("").apply(len)
    df["TotalSpend"] = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(0).sum(axis=1)
    return df

df = enrich(df)
df_test = enrich(df_test)

# 🧹 Подготовка X/y
drop_cols = ["Transported", "PassengerId", "Name", "Cabin"]
X = df.drop(columns=drop_cols)
y = df["Transported"]

# Категориальные и числовые признаки
cat_features = X.select_dtypes(include="object").columns.tolist()
num_features = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()

# Трансформер
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
    ("classifier", RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42))
])

# 📤 Train/Validation split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 🏋️ Обучение
model.fit(X_train, y_train)

# 🧪 Оценка
y_pred = model.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
print(f"Validation accuracy: {acc:.4f}")

# 🔮 Предсказания на тестовом
X_test = df_test.drop(columns=["PassengerId", "Name", "Cabin"])
X_test = enrich(X_test)
predictions = model.predict(X_test)

# 📁 Submission
submission["Transported"] = predictions.astype(bool)
submission.to_csv("submission.csv", index=False)
print("Готово: файл submission.csv сохранён!")
