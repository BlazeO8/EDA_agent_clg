
import numpy as np
import pandas as pd


def perform_eda(df: pd.DataFrame):
    """Performs basic Exploratory Data Analysis (EDA) on a given pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.
    """
    print("=" * 60)
    print(" 📊 EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 60)

    # 1. Basic Shape
    print(f"\n[1] DATASET DIMENSIONS")
    print(f"Number of Rows    : {df.shape[0]}")
    print(f"Number of Columns : {df.shape[1]}")

    # 2. Column Names and Data Types
    print(f"\n[2] COLUMNS & DATA TYPES")
    dtype_df = pd.DataFrame(
        {
            "Column Name": df.columns,
            "Data Type": df.dtypes.values,
            "Non-Null Count": df.notnull().sum().values,
        }
    )
    print(dtype_df.to_string(index=False))

    # 3. Missing Values Summary
    print(f"\n[3] MISSING VALUES ANALYSIS")
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)
    missing_df = pd.DataFrame(
        {"Missing Values": missing_count, "Percentage (%)": missing_pct}
    )
    missing_df = missing_df[missing_df["Missing Values"] > 0]

    if missing_df.empty:
        print("🎉 Great news! There are no missing values in this dataset.")
    else:
        print(missing_df.sort_values(by="Missing Values", ascending=False))

    # 4. Duplicate Rows
    print(f"\n[4] DUPLICATE ROWS")
    duplicate_count = df.duplicated().sum()
    duplicate_pct = (duplicate_count / len(df)) * 100
    print(
        f"Number of duplicate rows: {duplicate_count} ({duplicate_pct:.2f}% of total data)"
    )

    # 5. Numerical Summary & Skewness
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print(f"\n[5] NUMERICAL FEATURES SUMMARY")
        print(df[numerical_cols].describe().T)

        print(f"\n[6] NUMERICAL FEATURES SKEWNESS")
        skewness = df[numerical_cols].skew()
        skew_df = pd.DataFrame(
            {"Skewness": skewness, "Interpretation": np.where(skewness > 1, "Highly Right-Skewed", np.where(skewness < -1, "Highly Left-Skewed", "Approximately Symmetric"))}
        )
        print(skew_df)
    else:
        print(
            "\n[5 & 6] No numerical columns found to generate statistical summary."
        )

    # 7. Categorical Summary
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns
    if len(categorical_cols) > 0:
        print(f"\n[7] CATEGORICAL FEATURES SUMMARY")
        cat_summary = []
        for col in categorical_cols:
            cat_summary.append(
                {
                    "Column": col,
                    "Unique Values": df[col].nunique(),
                    "Most Frequent": df[col].mode()[0]
                    if not df[col].mode().empty
                    else np.nan,
                    "Frequency": df[col].value_counts().iloc[0]
                    if not df[col].empty
                    else 0,
                }
            )
        print(pd.DataFrame(cat_summary).to_string(index=False))
    else:
        print(
            "\n[7] No categorical columns found to generate frequency summary."
        )

    print("\n" + "=" * 60)
    print(" END OF EDA REPORT")
    print("=" * 60)
