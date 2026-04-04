# =====================================
# 1. Import libraries
# =====================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# =====================================
# 2. Load dataset
# =====================================
url = "titanic_survival.csv"
df = pd.read_csv(url)

print("First rows of dataset:")
print(df.head())

print("\nDataset shape:", df.shape)

# =====================================
# 3. Missing values
# =====================================
print("\nMissing values per column:")
print(df.isnull().sum())

# Count missing values in 'age'
missing_values_count = df['age'].isnull().sum()
print("\nMissing values in 'age':", missing_values_count)

# Count missing values in 'cabin'
print("Missing values in 'cabin':", df['cabin'].isnull().sum())

# =====================================
# 4. Handle missing data
# =====================================

# Drop rows where 'embarked' is missing
df = df.dropna(subset=['embarked'])

# Drop 'cabin' column
df = df.drop(columns=['cabin'])

# Fill missing 'age' with mean
df['age'].fillna(df['age'].mean(), inplace=True)

# Fill missing 'embarked' with most frequent value
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# =====================================
# 5. Encoding categorical variables
# =====================================

# One-hot encoding for 'embarked'
df = pd.get_dummies(df, columns=['embarked'])

# Label encoding for 'sex'
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])

# =====================================
# 6. Feature selection
# =====================================
df_selected = df[['pclass', 'sex', 'age', 'fare', 'survived']]

X = df_selected.drop('survived', axis=1)
y = df_selected['survived']

# =====================================
# 7. Train/Test split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)

# =====================================
# 8. Feature scaling
# =====================================

# StandardScaler
scaler_standard = StandardScaler()
X_train_std = scaler_standard.fit_transform(X_train)
X_test_std = scaler_standard.transform(X_test)

# MinMaxScaler
scaler_minmax = MinMaxScaler()
X_train_mm = scaler_minmax.fit_transform(X_train)
X_test_mm = scaler_minmax.transform(X_test)

print("\nStandardScaler sample:")
print(X_train_std[:5])

print("\nMinMaxScaler sample:")
print(X_train_mm[:5])