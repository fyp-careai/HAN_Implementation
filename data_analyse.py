import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("final_processed_data/merged_coop_ruhunu_patient_data.csv")

print("Dataset Shape:", df.shape)

# Convert date
df["record_date"] = pd.to_datetime(df["record_date"], errors='coerce')

# ==============================
# 2. Basic Statistics
# ==============================

print("\nUnique Patients:", df["patient_id"].nunique())
print("Unique Tests:", df["mapped_test_name"].nunique())

print("\nSex Distribution:")
print(df["sex"].value_counts())

print("\nAge Distribution:")
print(df["age"].describe())

# ==============================
# 3. Test Distribution
# ==============================

test_counts = df["mapped_test_name"].value_counts()

plt.figure(figsize=(12,6))
test_counts.head(20).plot(kind='bar')
plt.title("Top 20 Lab Tests")
plt.ylabel("Count")
plt.xticks(rotation=60)
plt.show()

# ==============================
# 4. Records Per Patient
# ==============================

records_per_patient = df.groupby("patient_id").size()

plt.figure(figsize=(8,5))
sns.histplot(records_per_patient, bins=50)
plt.title("Records per Patient Distribution")
plt.xlabel("Number of Tests")
plt.show()

print("\nRecords per Patient Stats:")
print(records_per_patient.describe())

# ==============================
# 5. Value Distribution per Test
# ==============================

top_tests = test_counts.head(5).index

for test in top_tests:
    
    plt.figure(figsize=(6,4))
    
    sns.histplot(
        df[df["mapped_test_name"] == test]["value"],
        bins=50,
        kde=True
    )
    
    plt.title(f"Value Distribution: {test}")
    plt.xlabel("Value")
    plt.show()

# ==============================
# 6. Outlier Detection
# ==============================

plt.figure(figsize=(12,6))
sns.boxplot(
    x="mapped_test_name",
    y="value",
    data=df[df["mapped_test_name"].isin(top_tests)]
)

plt.title("Outlier Detection for Top Tests")
plt.xticks(rotation=60)
plt.show()

# ==============================
# 7. Temporal Distribution
# ==============================

df["year"] = df["record_date"].dt.year

plt.figure(figsize=(8,5))
sns.countplot(x="year", data=df)
plt.title("Records by Year")
plt.show()

# ==============================
# 8. Test Range Statistics
# ==============================

range_stats = df.groupby("mapped_test_name")["value"].describe()

print("\nTest Value Ranges:")
print(range_stats)

# ==============================
# 9. Save Analysis Table
# ==============================

range_stats.to_csv("test_value_ranges.csv")

print("\nSaved test range statistics to test_value_ranges.csv")