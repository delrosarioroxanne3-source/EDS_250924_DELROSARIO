# %%
import os

OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Outputs folder ready:", OUTPUT_DIR)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Exp1.csv")

print("Dataset loaded successfully.")
print(df.head())


# =========================
# OUTPUT FOLDER (SAFE RESET)
# =========================
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# PIPELINE CLASS
# =========================
class CNCPipeline:

    def __init__(self, dataframe, output_dir):
        self.df = dataframe
        self.output_dir = output_dir

    def clean_data(self):
        self.df = self.df.drop_duplicates()
        self.df = self.df.fillna(self.df.mean(numeric_only=True))
        print("Data cleaned successfully.")

    def descriptive_statistics(self):
        print("\n===== DESCRIPTIVE STATISTICS =====")
        print(self.df.describe())

    def correlation_analysis(self):
        corr = self.df.select_dtypes(include=np.number).corr()

        print("\n===== CORRELATION MATRIX =====")
        print(corr)

        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, cmap="coolwarm")
        plt.title("Correlation Heatmap")

        plt.savefig(os.path.join(self.output_dir, "heatmap.png"), bbox_inches="tight")
        plt.close()


    def visualization(self):
        roughness_cols = ["Ra", "Rz", "Rt"]
        available_cols = [col for col in roughness_cols if col in self.df.columns]

        if len(available_cols) == 0:
            print("No roughness columns found.")
            return

        plt.figure(figsize=(10, 6))

        self.df[available_cols].plot(kind="box")

        plt.title("Surface Roughness Distribution Comparison (Ra, Rz, Rt)")
        plt.ylabel("Roughness Value")

        plt.savefig(os.path.join(self.output_dir, "roughness_boxplot.png"),
                    bbox_inches="tight")
        plt.close()

        print("Roughness comparison plot saved.")

    def run(self):
        self.clean_data()
        self.descriptive_statistics()
        self.correlation_analysis()
        self.visualization()

        print("\nPIPELINE FINISHED SUCCESSFULLY.")


# =========================
# RUN PIPELINE
# =========================
pipeline = CNCPipeline(df, OUTPUT_DIR)
pipeline.run()
# %%
import matplotlib.pyplot as plt
import os

# make sure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# choose columns safely
cols = ["Ra", "Rz", "Rt"]

for col in cols:
    if col in df.columns:

        data = df[col].dropna()

        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=20, edgecolor="black")

        plt.title(f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        plt.savefig(f"outputs/{col}_histogram.png", bbox_inches="tight")
        plt.show()

print("DONE: Visualizations created successfully.")
# %%
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

# 1. Fx vs Ra
if "Fx" in df.columns and "Ra" in df.columns:
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=df["Fx"], y=df["Ra"])
    plt.title("Cutting Force (Fx) vs Surface Roughness (Ra)")
    plt.xlabel("Fx")
    plt.ylabel("Ra")
    plt.savefig("outputs/Fx_vs_Ra.png", bbox_inches="tight")
    plt.show()

# 2. Fy vs Ra
if "Fy" in df.columns and "Ra" in df.columns:
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=df["Fy"], y=df["Ra"])
    plt.title("Cutting Force (Fy) vs Surface Roughness (Ra)")
    plt.xlabel("Fy")
    plt.ylabel("Ra")
    plt.savefig("outputs/Fy_vs_Ra.png", bbox_inches="tight")
    plt.show()

# 3. F vs Ra
if "F" in df.columns and "Ra" in df.columns:
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=df["F"], y=df["Ra"])
    plt.title("Total Force (F) vs Surface Roughness (Ra)")
    plt.xlabel("F")
    plt.ylabel("Ra")
    plt.savefig("outputs/F_vs_Ra.png", bbox_inches="tight")
    plt.show()

print("Correlation plots done.")
# %%
def correlation_analysis(self):

    corr = self.df.select_dtypes(include=np.number).corr()

    print("\n===== CORRELATION MATRIX =====")
    print(corr)

    # ensure folder exists every time (safe fix)
    os.makedirs(self.output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm")

    plt.title("Correlation Heatmap")

    save_path = os.path.join(self.output_dir, "heatmap.png")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()   # 🔥 THIS IS THE FIX FOR TRACEBACK

    print("Heatmap saved:", save_path)
# %%
import numpy as np

target = "Ra"

# =========================
# CLEAN DATA
# =========================
df_ml = df.copy()

for col in df_ml.columns:
    df_ml[col] = pd.to_numeric(df_ml[col], errors="coerce")

df_ml = df_ml.dropna(subset=[target])

X = df_ml[["Fx", "Fy", "F"]].fillna(0).values
y = df_ml[target].values

# =========================
# NORMALIZE
# =========================
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

X = (X - X_mean) / X_std

# add bias column
X = np.c_[np.ones(X.shape[0]), X]

# =========================
# TRAIN (NORMAL EQUATION)
# =========================
theta = np.linalg.inv(X.T @ X) @ X.T @ y

print("Model trained successfully.")
# %%
y_pred = X @ theta

mae = np.mean(np.abs(y - y_pred))
ss_total = np.sum((y - np.mean(y))**2)
ss_res = np.sum((y - y_pred)**2)
r2 = 1 - (ss_res / ss_total)

print("MAE:", mae)
print("R2 Score:", r2)
# %%
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(6,6))
plt.scatter(y, y_pred, alpha=0.6)

plt.xlabel("Actual Ra")
plt.ylabel("Predicted Ra")
plt.title("Actual vs Predicted (Manual Linear Regression)")

plt.plot([min(y), max(y)], [min(y), max(y)])

plt.savefig("outputs/manual_prediction.png", bbox_inches="tight")
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# PREDICTIONS
# =========================
y_pred = X @ theta

# =========================
# METRICS
# =========================
mae = np.mean(np.abs(y - y_pred))

ss_total = np.sum((y - np.mean(y))**2)
ss_res = np.sum((y - y_pred)**2)
r2 = 1 - (ss_res / ss_total)

print("===== MODEL PERFORMANCE =====")
print("MAE:", mae)
print("R2 Score:", r2)
# %%

import pandas as pd
import numpy as np
from scipy.stats import skew

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("your_dataset.csv")  # change filename

# Keep only numeric columns for safety
num_df = df.select_dtypes(include=np.number)

# -----------------------------
# 1. DESCRIPTIVE STATISTICS
# -----------------------------
print("\n===== MEAN =====")
print(num_df.mean())

print("\n===== MEDIAN =====")
print(num_df.median())

print("\n===== VARIANCE =====")
print(num_df.var())

print("\n===== STANDARD DEVIATION =====")
print(num_df.std())

# -----------------------------
# 2. SKEWNESS (NO SCIPY VERSION)
# -----------------------------
print("\n===== SKEWNESS =====")

def compute_skew(x):
    x = x.dropna()
    mean = x.mean()
    std = x.std()
    n = len(x)
    return ((n / ((n-1)*(n-2))) * np.sum(((x - mean) / std) ** 3))

skewness = num_df.apply(compute_skew)
print(skewness)

# -----------------------------
# 4. OUTLIER DETECTION (IQR METHOD)
# -----------------------------
def detect_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR)))

print("\n===== OUTLIER SUMMARY =====")
for col in num_df.columns:
    outliers = detect_outliers(num_df[col])
    print(f"{col}: {outliers.sum()} outliers")

# -----------------------------
# 5. COMPARATIVE ANALYSIS (FIXED)
# -----------------------------

# Ensure no missing values in F
df_clean = df.dropna(subset=["F"])

# Create Force groups safely
df_clean["Force_Group"] = np.where(
    df_clean["F"] > df_clean["F"].median(),
    "High Force",
    "Low Force"
)

comparison = df_clean.groupby("Force_Group")[["Ra", "Rz", "Rt", "Fx", "Fy", "Fz", "F"]].mean()

print("\n===== COMPARATIVE ANALYSIS =====")
print(comparison)
# %%
import pandas as pd
import matplotlib.pyplot as plt

# LOAD DATASET
df = pd.read_csv("exp1.csv")

# DISPLAY FIRST 5 ROWS
print(df.head())

# DESCRIPTIVE STATISTICS
print("\nDescriptive Statistics:")
print(df.describe())

# CHECK MISSING VALUES
print("\nMissing Values:")
print(df.isnull().sum())

# HISTOGRAM
df.hist(figsize=(10,8))
plt.show()
# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Exp1.csv")

# Features and target
X = df[["vc", "f", "ap"]]
y = df["Ra"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Reset index for clean plotting
y_test = y_test.reset_index(drop=True)

# ===== COMPARATIVE BAR GRAPH (BEST FOR IEEE) =====
plt.figure(figsize=(12,6))

index = range(len(y_test))

plt.bar(index, y_test, width=0.4, label="Actual Ra")
plt.bar(index, y_pred, width=0.4, label="Predicted Ra", alpha=0.7)

plt.title("Comparative Analysis: Actual vs Predicted Surface Roughness (Ra)")
plt.xlabel("Sample Index")
plt.ylabel("Surface Roughness (Ra)")
plt.legend()
plt.grid(True)

plt.show()
# %%
