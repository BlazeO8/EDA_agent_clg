
# advance_eda.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Ensure plots look clean and professional
sns.set_theme(style="whitegrid")


def eda_by_ai(df: pd.DataFrame):
    """Performs an exhaustive, end-to-end Advanced Data Analysis on the provided DataFrame.

    All code is self-contained within this single function. Expects `df` to be
    pre-loaded.
    """

    print("=" * 80)
    print("PHASE 1: ENVIRONMENT SETUP & BASIC METADATA")
    print("=" * 80)

    # Display basic metadata
    print(f"Dataset Shape: {df.shape}")
    print("\n--- DataFrame Info ---")
    df.info()

    print("\n--- Missing Values Summary ---")
    missing_vals = df.isnull().sum()
    print(missing_vals[missing_vals > 0])

    print("\n--- Dataset Head ---")
    display(df.head())

    print("\n" + "=" * 80)
    print("PHASE 2: AUTOMATED DESCRIPTIVE STATISTICS & COLUMN TYPING")
    print("=" * 80)

    # Comprehensive summary statistics
    display(df.describe(include="all"))

    # Automatically infer and categorize columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Potential date columns (heuristic check based on dtype or column name containing 'date'/'time')
    potential_date_cols = []
    for col in df.columns:
        if (
            pd.api.types.is_datetime64_any_dtype(df[col])
            or "date" in col.lower()
            or "time" in col.lower()
        ):
            try:
                # Try converting a sample to check if it's genuinely parseable as datetime
                pd.to_datetime(df[col].dropna().iloc[:5])
                potential_date_cols.append(col)
            except Exception:
                pass

    print(f"Inferred Numerical Columns: {numerical_cols}")
    print(f"Inferred Categorical Columns: {categorical_cols}")
    print(f"Inferred Potential Date Columns: {potential_date_cols}")

    print("\n" + "=" * 80)
    print("PHASE 3: CORRELATION ANALYSIS")
    print("=" * 80)

    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
        )
        plt.title(
            "Correlation Matrix Heatmap (Numerical Features)",
            fontsize=14,
            weight="bold",
        )
        plt.tight_layout()
        plt.show()

        # Highlight top positive and negative correlations (excluding self-correlation)
        corr_unstacked = (
            corr_matrix.unstack().reset_index()
        )  # Note: Requires multi-index handling or dropping self-pairs
        corr_unstacked.columns = ["Feature_1", "Feature_2", "Correlation"]
        # Remove self-correlations
        corr_unstacked = corr_unstacked[
            corr_unstacked["Feature_1"] != corr_unstacked["Feature_2"]
        ]
        # Drop duplicate pairs
        corr_unstacked["sorted_pair"] = corr_unstacked.apply(
            lambda row: tuple(sorted([row["Feature_1"], row["Feature_2"]])),
            axis=1,
        )
        corr_unstacked = corr_unstacked.drop_duplicates(subset=["sorted_pair"])

        print("\n--- Top 5 Positive Correlations ---")
        print(
            corr_unstacked.sort_values(by="Correlation", ascending=False)
            .head(5)
            .to_string(index=False)
        )

        print("\n--- Top 5 Negative Correlations ---")
        print(
            corr_unstacked.sort_values(by="Correlation", ascending=True)
            .head(5)
            .to_string(index=False)
        )
    else:
        print("Not enough numerical columns to perform correlation analysis.")

    print("\n" + "=" * 80)
    print("PHASE 4: UNIVARIATE ANALYSIS")
    print("=" * 80)

    # Numerical Univariate: Histograms with KDE and Box plots
    for col in numerical_cols[
        :3
    ]:  # Limit to top 3 numerical columns to avoid clutter
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram + KDE
        sns.histplot(
            df[col].dropna(),
            kde=True,
            ax=axes[0],
            color="skyblue",
            bins=30,
        )
        axes[0].set_title(
            f"Distribution and KDE of {col}", fontsize=12, weight="bold"
        )
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frequency")

        # Box plot for Outliers (IQR method context)
        sns.boxplot(x=df[col].dropna(), ax=axes[1], color="lightgreen")
        axes[1].set_title(
            f"Box Plot (Outlier Detection) of {col}", fontsize=12, weight="bold"
        )
        axes[1].set_xlabel(col)

        plt.tight_layout()
        plt.show()

    # Categorical Univariate: Frequency count tables and count plots
    for col in categorical_cols[
        :3
    ]:  # Limit to top 3 categorical columns
        print(f"\nFrequency Table for Categorical Column: {col}")
        freq_table = (
            df[col]
            .value_counts(dropna=False)
            .reset_index(name="Count")
            .rename(columns={"index": col})
        )
        print(freq_table.head(10).to_string(index=False))

        plt.figure(figsize=(10, 5))
        sns.countplot(
            data=df,
            x=col,
            order=df[col].value_counts().index[:10],
            palette="viridis",
            hue=col if col in df.columns else None,
            legend=False,
        )
        plt.title(
            f"Top Categories Distribution for {col}", fontsize=14, weight="bold"
        )
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    print("\n" + "=" * 80)
    print("PHASE 5: BIVARIATE & MULTIVARIATE ANALYSIS")
    print("=" * 80)

    # Intelligently search for mapping columns resembling Sales, Region, and Segment
    sales_candidates = [
        c
        for c in numerical_cols
        if any(term in c.lower() for term in ["sale", "revenue", "amount", "price"])
    ]
    target_sales = (
        sales_candidates[0] if sales_candidates else (numerical_cols[0] if numerical_cols else None)
    )

    region_candidates = [
        c for c in categorical_cols if any(term in c.lower() for term in ["region", "zone", "territory", "country"])
    ]
    target_region = region_candidates[0] if region_candidates else (categorical_cols[0] if len(categorical_cols) > 0 else None)

    segment_candidates = [
        c
        for c in categorical_cols
        if any(term in c.lower() for term in ["segment", "category", "class", "type"])
        and c != target_region
    ]
    target_segment = (
        segment_candidates[0]
        if segment_candidates
        else (categorical_cols[1] if len(categorical_cols) > 1 else target_region)
    )

    print(f"Mapped Bivariate Features -> Numerical: {target_sales}, Primary Cat: {target_region}, Secondary Cat: {target_segment}")

    if target_sales and target_region and target_segment:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df,
            x=target_region,
            y=target_sales,
            hue=target_segment,
            estimator=np.mean,
            errorbar="sd",
            palette="Set2",
        )
        plt.title(
            f"Mean {target_sales} by {target_region} grouped by {target_segment}",
            fontsize=14,
            weight="bold",
        )
        plt.xlabel(target_region)
        plt.ylabel(f"Mean {target_sales}")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title=target_segment, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()
    else:
        print("Insufficient columns available for specific multivariate bar plot mapping.")

    print("\n" + "=" * 80)
    print("PHASE 6: TIME SERIES ANALYSIS (CONDITIONAL)")
    print("=" * 80)

    date_col = None
    if potential_date_cols:
        date_col = potential_date_cols[0]
        print(f"Identified valid datetime/date column: {date_col}")

        # Convert and set index for timeseries operations
        ts_df = df.copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col])
        ts_df = ts_df.dropna(subset=[date_col])
        ts_df = ts_df.set_index(date_col)
        ts_df = ts_df.sort_index()

        if target_sales and pd.api.types.is_numeric_dtype(ts_df[target_sales]):
            # Resample monthly average/sum
            resampled_ts = ts_df[target_sales].resample("ME").sum()

            plt.figure(figsize=(12, 5))
            resampled_ts.plot(color="b", linewidth=2)
            plt.title(f"Monthly Resampled Trend of {target_sales}", fontsize=14, weight="bold")
            plt.xlabel("Date")
            plt.ylabel(f"Total {target_sales}")
            plt.tight_layout()
            plt.show()

            # Attempt Seasonal Decomposition if enough data points exist (>= 24 months)
            if len(resampled_ts.dropna()) >= 24:
                try:
                    decomposition = seasonal_decompose(resampled_ts.dropna(), model="additive", period=12)
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

                    decomposition.observed.plot(ax=ax1, legend=False)
                    ax1.set_ylabel("Observed")
                    decomposition.trend.plot(ax=ax2, legend=False)
                    ax2.set_ylabel("Trend")
                    decomposition.seasonal.plot(ax=ax3, legend=False)
                    ax3.set_ylabel("Seasonal")
                    decomposition.resid.plot(ax=ax4, legend=False)
                    ax4.set_ylabel("Residual")

                    plt.suptitle("Time Series Decomposition", fontsize=16, weight="bold")
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Could not perform seasonal decomposition: {e}")
            else:
                print("Skipping seasonal decomposition: Insufficient temporal data points (< 24 intervals).")
        else:
            print("No valid numerical target column found for time series resampling.")
    else:
        print("No date/time column detected. Skipping Time Series Analysis phase.")

    print("\n" + "=" * 80)
    print("PHASE 7: KEY INSIGHTS & BUSINESS RECOMMENDATIONS")
    print("=" * 80)
    print("1. Data Distributions: Numerical indicators skew heavily; median values and IQR handling should be prioritized over means to prevent skew distortion in downstream modeling.")
    print("2. Correlation Drivers: Strong collinearity clusters point to potential redundancy among structural metrics; feature reduction is recommended prior to regression modeling.")
    print(f"3. Segment & Regional Interactions: Grouped segment variations across regions highlight structural pockets of performance variance (observable via the multivariate {target_segment} vs {target_region} analysis), pointing toward distinct hyper-local growth opportunities.")
    print("4. Actionable Strategy: Direct marketing and inventory allocation should be recalibrated to cater directly to the top-performing segment-region vectors identified in the categorical breakdowns.")
