
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

# Загрузка данных
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")

# Целевая переменная
df["Transported"] = df["Transported"].astype(int)

# Feature Engineering
def enrich(df):
    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    df["Group"] = df["PassengerId"].apply(lambda x: x.split("_")[0])
    df["NameLength"] = df["Name"].fillna("").apply(len)
    df["TotalSpend"] = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(0).sum(axis=1)
    df["IsZeroSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["CryoZero"] = ((df["CryoSleep"] == True) & (df["TotalSpend"] == 0)).astype(int)
    df["GroupSize"] = df["Group"].map(df["Group"].value_counts())
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)
    df["SideIsP"] = (df["Side"] == "P").astype(int)
    return df

df = enrich(df)
df_test = enrich(df_test)

# Подготовка признаков и таргета
drop_cols = ["Transported", "PassengerId", "Name", "Cabin"]
X = df.drop(columns=drop_cols)
y = df["Transported"]
X_test = df_test.drop(columns=["PassengerId", "Name", "Cabin"], errors="ignore")

# Обработка категориальных
cat_features = X.select_dtypes(include="object").columns.tolist()
for col in cat_features:
    X[col] = X[col].astype(str).fillna("missing")
    X_test[col] = X_test[col].astype(str).fillna("missing")

# GroupKFold сплит
gkf = GroupKFold(n_splits=5)
for train_idx, valid_idx in gkf.split(X, y, groups=df["Group"]):
    X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    break

# Удаляем 'Group' везде
for df_ in [X_train, X_valid, X_test]:
    if "Group" in df_.columns:
        df_.drop(columns=["Group"], inplace=True)

# Выравниваем порядок колонок
X_test = X_test[X_train.columns]

# CatBoostClassifier с early stopping
model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    cat_features=cat_features,
    early_stopping_rounds=50,
    verbose=100,
    random_state=42
)

model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

# Оценка
y_pred = model.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
print(f"Validation accuracy (CatBoost): {acc:.4f}")

# Предсказание
predictions = model.predict(X_test)
submission["Transported"] = predictions.astype(bool)
submission.to_csv("submission_catboost_final.csv", index=False)
print("✅ submission_catboost_final.csv сохранён")
