import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def build_preprocessor(df: pd.DataFrame, target: str):
  X = df.drop(columns=[target])
  num = X.select_dtypes(include="number").columns.tolist()
  cat = X.select_dtypes(exclude="number").columns.tolist()
  num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
  cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])
  return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])
