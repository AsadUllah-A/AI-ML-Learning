import pandas as pd
import numpy as np

data = {
    "StudentID": [1, 2, 3, 4, 5, 5],
    "Name": ["Asad", "Ali", "Ahmed", "Usman", "Hassan", "Hassan"],
    "Department": ["SE", "CS", "SE", "AI", "CS", "CS"],
    "Age": [22, 21, np.nan, 20, 24, 24],
    "Marks": [85, 45, 72, np.nan, 60, 60]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

print("\nMissing Values:")
print(df.isnull().sum())

df = df.drop_duplicates()

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())

df["Result"] = np.where(df["Marks"] >= 50, "Pass", "Fail")

print("\nCleaned Data:")
print(df)

department_summary = df.groupby("Department")["Marks"].agg(["mean", "max", "min", "count"])

print("\nDepartment Summary:")
print(department_summary)

df = df.sort_values("Marks", ascending=True)

print("\nSorted Data:")
print(df)

df.to_csv("cleaned_students.csv", index=False)

print("\nCleaned data saved successfully.")