"""
Titanic Dataset Analysis — Pandas Showcase
Covers: loading, inspection, cleaning, selection, aggregation,
        groupby, merging, reshaping, time-series basics, and export.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ── 1. LOAD ──────────────────────────────────────────────────────────────────

df = sns.load_dataset("titanic")          # ~891 rows; seaborn bundles this CSV
print("Shape:", df.shape)                 # (rows, cols)


# ── 2. INSPECT ───────────────────────────────────────────────────────────────

print("\n--- dtypes ---")
print(df.dtypes)

print("\n--- first 5 rows ---")
print(df.head())

print("\n--- descriptive stats (numeric) ---")
print(df.describe())

print("\n--- descriptive stats (categorical) ---")
print(df.describe(include="object"))

print("\n--- missing values ---")
print(df.isnull().sum())


# ── 3. SELECTION ─────────────────────────────────────────────────────────────

# Column selection
ages = df["age"]                          # Series
subset = df[["survived", "pclass", "age", "sex", "fare"]]

# Boolean / conditional filtering
survived = df[df["survived"] == 1]
first_class_women = df[(df["pclass"] == 1) & (df["sex"] == "female")]

# Label-based and integer-based indexing
row_by_label = df.loc[0, ["sex", "age", "survived"]]     # loc: label
row_by_pos   = df.iloc[0, [1, 2, 3]]                     # iloc: position

# query() — SQL-like syntax
cheap_survivors = df.query("survived == 1 and fare < 20")

print("\nFirst-class female survivors:", len(first_class_women[first_class_women["survived"] == 1]))


# ── 4. CLEANING ──────────────────────────────────────────────────────────────

df_clean = df.copy()

# Fill missing age with median; drop rows still missing 'embarked'
df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())
df_clean.dropna(subset=["embarked"], inplace=True)

# Drop a column that is mostly NaN
df_clean.drop(columns=["deck"], inplace=True)

# Rename columns
df_clean.rename(columns={"pclass": "passenger_class", "sibsp": "siblings_spouses"}, inplace=True)

# Cast types
df_clean["survived"] = df_clean["survived"].astype(bool)

# Remove duplicates (none here, but demonstrates the pattern)
df_clean.drop_duplicates(inplace=True)

print("\nClean shape:", df_clean.shape)


# ── 5. FEATURE ENGINEERING ───────────────────────────────────────────────────

df_clean["family_size"] = df_clean["siblings_spouses"] + df_clean["parch"] + 1
df_clean["is_alone"] = df_clean["family_size"] == 1

df_clean["age_band"] = pd.cut(
    df_clean["age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["child", "teen", "young adult", "adult", "senior"],
)

df_clean["fare_log"] = df_clean["fare"].apply(lambda x: x ** 0.5)   # apply()


# ── 6. AGGREGATION ───────────────────────────────────────────────────────────

print("\n--- survival rate by sex ---")
print(df_clean.groupby("sex")["survived"].mean().round(3))

print("\n--- mean fare & age by class ---")
print(df_clean.groupby("passenger_class")[["fare", "age"]].mean().round(2))

# agg() for multiple functions at once
print("\n--- fare stats by class ---")
print(df_clean.groupby("passenger_class")["fare"].agg(["min", "mean", "median", "max"]).round(2))

# named aggregations (pandas ≥ 0.25)
summary = df_clean.groupby("passenger_class").agg(
    total_passengers=("survived", "count"),
    survivors=("survived", "sum"),
    avg_age=("age", "mean"),
    avg_fare=("fare", "mean"),
).round(2)
print("\n--- class summary ---")
print(summary)

# transform() — broadcast group stat back to original index
df_clean["class_mean_fare"] = df_clean.groupby("passenger_class")["fare"].transform("mean")

# value_counts() — frequency of a categorical column
print("\n--- embarkation counts ---")
print(df_clean["embarked"].value_counts())


# ── 7. PIVOT / RESHAPE ───────────────────────────────────────────────────────

pivot = df_clean.pivot_table(
    values="survived",
    index="passenger_class",
    columns="sex",
    aggfunc="mean",
).round(3)
print("\n--- survival rate pivot (class × sex) ---")
print(pivot)

# melt() — wide → long format
pivot_reset = pivot.reset_index()
melted = pivot_reset.melt(id_vars="passenger_class", var_name="sex", value_name="survival_rate")
print("\n--- melted (long format) ---")
print(melted.head())

# crosstab — quick frequency table
ct = pd.crosstab(df_clean["passenger_class"], df_clean["survived"], normalize="index").round(3)
print("\n--- crosstab: class vs survived (row %) ---")
print(ct)


# ── 8. SORTING & RANKING ─────────────────────────────────────────────────────

top_fares = df_clean.nlargest(5, "fare")[["who", "passenger_class", "fare"]]
print("\n--- top 5 fares ---")
print(top_fares)

df_clean["fare_rank"] = df_clean["fare"].rank(ascending=False, method="min")


# ── 9. MERGE / JOIN ──────────────────────────────────────────────────────────

# Build a small lookup table for embarkation port names
port_names = pd.DataFrame({
    "embarked": ["S", "C", "Q"],
    "port": ["Southampton", "Cherbourg", "Queenstown"],
})

df_merged = df_clean.merge(port_names, on="embarked", how="left")
print("\nPort column added:", "port" in df_merged.columns)


# ── 10. STRING OPERATIONS ────────────────────────────────────────────────────

df_merged["who_upper"] = df_merged["who"].str.upper()
df_merged["starts_with_m"] = df_merged["who"].str.startswith("m")


# ── 11. WINDOW FUNCTIONS ─────────────────────────────────────────────────────

# Rolling mean on fare (sorted by index)
df_merged_sorted = df_merged.sort_values("fare").reset_index(drop=True)
df_merged_sorted["fare_rolling_mean"] = df_merged_sorted["fare"].rolling(window=10, min_periods=1).mean()
df_merged_sorted["fare_cumsum"] = df_merged_sorted["fare"].cumsum()


# ── 12. VISUALISATION ────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Titanic — Pandas Analysis", fontsize=14)

# Survival rate by class & sex
pivot.plot(kind="bar", ax=axes[0, 0], colormap="Set2", rot=0)
axes[0, 0].set_title("Survival rate by class & sex")
axes[0, 0].set_ylabel("Rate")

# Age distribution
df_clean["age"].plot(kind="hist", bins=30, ax=axes[0, 1], color="steelblue", edgecolor="white")
axes[0, 1].set_title("Age distribution")
axes[0, 1].set_xlabel("Age")

# Fare by class (box plot)
df_clean.boxplot(column="fare", by="passenger_class", ax=axes[1, 0])
axes[1, 0].set_title("Fare by class")
axes[1, 0].set_xlabel("Class")
plt.sca(axes[1, 0])
plt.title("Fare by class")

# Family size vs survival
family_survival = df_clean.groupby("family_size")["survived"].mean()
family_survival.plot(kind="bar", ax=axes[1, 1], color="coral", rot=0)
axes[1, 1].set_title("Survival rate by family size")
axes[1, 1].set_xlabel("Family size")
axes[1, 1].set_ylabel("Rate")

plt.tight_layout()
plt.savefig("titanic_analysis.png", dpi=120)
plt.show()
print("\nPlot saved to titanic_analysis.png")


# ── 13. EXPORT ───────────────────────────────────────────────────────────────

summary.to_csv("class_summary.csv")
summary.to_json("class_summary.json", orient="index", indent=2)
print("Exported class_summary.csv and class_summary.json")

print("\nDone.")
