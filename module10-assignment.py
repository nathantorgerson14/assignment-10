
# Module 10 Assignment: Data Manipulation and Cleaning with Pandas
# UrbanStyle Customer Data Cleaning

import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

print("=" * 60)
print("URBANSTYLE CUSTOMER DATA CLEANING")
print("=" * 60)

csv_content = """customer_id,first_name,last_name,email,phone,join_date,last_purchase,total_purchases,total_spent,preferred_category,satisfaction_rating,age,city,state,loyalty_status
CS001,John,Smith,johnsmith@email.com,(555) 123-4567,2023-01-15,2023-12-01,12,"1,250.99",Menswear,4.5,35,Tampa,FL,Gold
CS002,Emily,Johnson,emily.j@email.com,555.987.6543,01/25/2023,10/15/2023,8,$875.50,Womenswear,4,28,Miami,FL,Silver
CS003,Michael,Williams,mw@email.com,(555)456-7890,2023-02-10,2023-11-20,15,"2,100.75",Footwear,5,42,Orlando,FL,Gold
CS004,JESSICA,BROWN,jess.brown@email.com,5551234567,2023-03-05,2023-12-10,6,659.25,Womenswear,3.5,31,Tampa,FL,Bronze
CS005,David,jones,djones@email.com,555-789-1234,2023-03-20,2023-09-18,4,350.00,Menswear,,45,Jacksonville,FL,Bronze
CS006,Sarah,Miller,sarah_miller@email.com,(555) 234-5678,2023-04-12,2023-12-05,10,1450.30,Accessories,4,29,Tampa,FL,Silver
CS007,Robert,Davis,robert.davis@email.com,555.444.7777,04/30/2023,11/25/2023,7,$725.80,Footwear,4.5,38,Miami,FL,Silver
CS008,Jennifer,Garcia,jen.garcia@email.com,(555)876-5432,2023-05-15,2023-10-30,3,280.50,ACCESSORIES,3,25,Orlando,FL,Bronze
CS009,Michael,Williams,m.williams@email.com,5558889999,2023-06-01,2023-12-07,9,1100.00,Menswear,4,39,Jacksonville,FL,Silver
CS010,Emily,Johnson,emilyjohnson@email.com,555-321-6547,2023-06-15,2023-12-15,14,"1,875.25",Womenswear,4.5,27,Miami,FL,Gold
CS006,Sarah,Miller,sarah_miller@email.com,(555) 234-5678,2023-04-12,2023-12-05,10,1450.30,Accessories,4,29,Tampa,FL,Silver
CS011,Amanda,,amanda.p@email.com,(555) 741-8529,2023-07-10,,2,180.00,womenswear,3,32,Tampa,FL,Bronze
CS012,Thomas,Wilson,thomas.w@email.com,,2023-07-25,2023-11-02,5,450.75,menswear,4,44,Orlando,FL,Bronze
CS013,Lisa,Anderson,lisa.a@email.com,555.159.7530,08/05/2023,,0,0.00,Womenswear,,30,Miami,FL,
CS014,James,Taylor,jtaylor@email.com,555-951-7530,2023-08-20,2023-10-10,11,"1,520.65",Footwear,4.5,,Jacksonville,FL,Gold
CS015,Karen,Thomas,karen.t@email.com,(555) 357-9512,2023-09-05,2023-12-12,6,685.30,Womenswear,4,36,Tampa,FL,Silver
"""

customer_data_csv = StringIO(csv_content)

# TODO 1: Load and Explore the Dataset
raw_df = pd.read_csv(customer_data_csv)
print("\nInitial Data Preview:")
print(raw_df.head())
print("\nInitial Data Info:")
print(raw_df.info())

initial_missing_counts = raw_df.isna().sum()
initial_duplicate_count = int(raw_df.duplicated().sum())

# TODO 2: Handle Missing Values
missing_value_report = raw_df.isna().sum()

satisfaction_median = float(raw_df["satisfaction_rating"].median())
df_missing_handled = raw_df.copy()
df_missing_handled["satisfaction_rating"] = df_missing_handled["satisfaction_rating"].fillna(satisfaction_median)

date_fill_strategy = "forward_fill"
df_missing_handled["last_purchase"] = df_missing_handled["last_purchase"].ffill()

df_missing_handled["last_name"] = df_missing_handled["last_name"].fillna("Unknown")
df_missing_handled["phone"] = df_missing_handled["phone"].fillna("0000000000")
df_missing_handled["age"] = df_missing_handled["age"].fillna(raw_df["age"].median())
df_missing_handled["loyalty_status"] = df_missing_handled["loyalty_status"].fillna("Bronze")

df_no_missing = df_missing_handled.copy()

# TODO 3: Correct Data Types
df_typed = df_no_missing.copy()

df_typed["join_date"] = pd.to_datetime(df_typed["join_date"], format="mixed", errors="coerce")
df_typed["last_purchase"] = pd.to_datetime(df_typed["last_purchase"], format="mixed", errors="coerce")

df_typed["total_spent"] = pd.to_numeric(
    df_typed["total_spent"].astype(str).str.replace(r"[\$,]", "", regex=True),
    errors="coerce"
)

df_typed["total_purchases"] = pd.to_numeric(df_typed["total_purchases"], errors="coerce").astype(int)
df_typed["age"] = pd.to_numeric(df_typed["age"], errors="coerce").astype(int)
df_typed["satisfaction_rating"] = pd.to_numeric(df_typed["satisfaction_rating"], errors="coerce")

# TODO 4: Clean and Standardize Text Data
df_text_cleaned = df_typed.copy()

df_text_cleaned["first_name"] = df_text_cleaned["first_name"].str.strip().str.title()
df_text_cleaned["last_name"] = df_text_cleaned["last_name"].str.strip().str.title()
df_text_cleaned["preferred_category"] = df_text_cleaned["preferred_category"].str.strip().str.title()
df_text_cleaned["city"] = df_text_cleaned["city"].str.strip().str.title()
df_text_cleaned["state"] = df_text_cleaned["state"].str.strip().str.upper()
df_text_cleaned["loyalty_status"] = df_text_cleaned["loyalty_status"].str.strip().str.title()
df_text_cleaned["email"] = df_text_cleaned["email"].str.strip().str.lower()

phone_format = "(###) ###-####"

def format_phone(phone):
    digits = "".join(filter(str.isdigit, str(phone)))
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return "(000) 000-0000"

df_text_cleaned["phone"] = df_text_cleaned["phone"].apply(format_phone)

# TODO 5: Remove Duplicates
duplicate_count = int(df_text_cleaned.duplicated().sum())
df_no_duplicates = df_text_cleaned.drop_duplicates(keep="first").copy()

# TODO 6: Add Derived Features
reference_date = df_no_duplicates["last_purchase"].max()
df_no_duplicates["days_since_last_purchase"] = (
    reference_date - df_no_duplicates["last_purchase"]
).dt.days

df_no_duplicates["average_purchase_value"] = np.where(
    df_no_duplicates["total_purchases"] > 0,
    df_no_duplicates["total_spent"] / df_no_duplicates["total_purchases"],
    0
)

df_no_duplicates["purchase_frequency_category"] = np.select(
    [
        df_no_duplicates["total_purchases"] >= 10,
        df_no_duplicates["total_purchases"].between(5, 9)
    ],
    ["High", "Medium"],
    default="Low"
)

# TODO 7: Clean Up the DataFrame
df_renamed = df_no_duplicates.rename(columns={
    "email": "email_address",
    "phone": "phone_number"
})

df_final = df_renamed.drop(columns=[]).copy()
df_final = df_final.sort_values(by="total_spent", ascending=False).reset_index(drop=True)

# TODO 8: Generate Insights from Cleaned Data
avg_spent_by_loyalty = df_final.groupby("loyalty_status")["total_spent"].mean().sort_values(ascending=False)

category_revenue = df_final.groupby("preferred_category")["total_spent"].sum().sort_values(ascending=False)

satisfaction_spend_corr = float(df_final["satisfaction_rating"].corr(df_final["total_spent"]))

# TODO 9: Generate Final Report
print("\n" + "=" * 60)
print("URBANSTYLE CUSTOMER DATA CLEANING REPORT")
print("=" * 60)

total_missing_entries = int(initial_missing_counts.sum())
data_type_issues = [
    "Mixed date formats in join_date and last_purchase",
    "Currency symbols/commas in total_spent",
    "Inconsistent numeric types in age and total_purchases"
]

print("Data Quality Issues:")
print(f"- Missing Values: {total_missing_entries} total missing entries")
print(f"- Duplicates: {initial_duplicate_count} duplicate records found")
print(f"- Data Type Issues: {data_type_issues}")

print("\nStandardization Changes:")
print("- Names: Converted to proper case")
print("- Categories: Standardized to title case (Menswear, Womenswear, Footwear, Accessories)")
print(f"- Phone Numbers: Standardized to {phone_format}")

print("\nKey Business Insights:")
print(f"- Customer Base: {len(df_final)} total customers")
print("- Revenue by Loyalty:")
print(avg_spent_by_loyalty.round(2))
print(f"- Top Category: {category_revenue.idxmax()} with ${category_revenue.max():,.2f}")
print(f"- Satisfaction/Spend Correlation: {satisfaction_spend_corr:.3f}")

print("\nFirst 5 Rows of Cleaned Dataset:")
print(df_final.head())
