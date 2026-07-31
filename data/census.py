"""
Adult / Census Income — data preparation for fairness analysis.

Each step is action + justification (methodology for the fairness audit).
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
RAW_CSV = DATA_DIR / "adult.csv"
CLEAN_CSV = DATA_DIR / "adult_clean.csv"

PROTECTED_ATTRIBUTE = "gender"  # race audited in a later pass
OUTCOME = "income"


def step_1_load_and_inspect(path: Path) -> pd.DataFrame:
    """Step 1 — Load the CSV and inspect before altering anything."""
    df = pd.read_csv(path)
    print("=" * 72)
    print("STEP 1 — Load and inspect")
    print("=" * 72)
    print(
        "Justification: Inspecting before altering is good practice; "
        "cleaning choices cannot be justified if made blind.\n"
    )
    print(f"df.shape: {df.shape}")
    print("\ndf.head():")
    print(df.head())
    print("\ndf.info():")
    df.info()
    print("\ndf.describe():")
    print(df.describe())
    return df


def step_2_identify_key_columns(df: pd.DataFrame) -> None:
    """Step 2 — Understand columns; name protected attribute and outcome."""
    print("\n" + "=" * 72)
    print("STEP 2 — Column roles")
    print("=" * 72)
    print(
        "Justification: Document which columns play which role in the "
        "fairness analysis so the reader knows what is audited against what.\n"
    )
    roles = {
        "x": "Row identifier (no person-level meaning)",
        "age": "Numeric feature — age in years",
        "workclass": "Categorical feature — type of employer",
        "education": "Categorical feature — highest education level",
        "marital-status": "Categorical feature — marital status",
        "relationship": "Categorical feature — household relationship",
        "race": "Protected attribute (secondary audit, later)",
        "gender": "Protected attribute (primary audit)",
        "hours-per-week": "Numeric feature — hours worked per week",
        "income": "Outcome — income band (<=50K / >50K)",
    }
    for col in df.columns:
        print(f"  • {col}: {roles.get(col, '(see dataset docs)')}")
    print(f"\nKey columns for this project:")
    print(f"  Protected attribute → {PROTECTED_ATTRIBUTE}")
    print(f"  Outcome             → {OUTCOME}")


def step_3_find_missing_values(df: pd.DataFrame) -> pd.Series:
    """Step 3 — Count '?' placeholders (not caught by isna() alone)."""
    print("\n" + "=" * 72)
    print("STEP 3 — Find missing values encoded as '?'")
    print("=" * 72)
    print(
        "Justification: This dataset hides missingness as the string '?', "
        "so a normal missing-value check misses them. Counting '?' shows "
        "the data was not assumed clean.\n"
    )
    question_counts = (df.astype(str) == "?").sum()
    print(question_counts)
    print(f"\nTotal cells with '?': {int(question_counts.sum())}")
    print(f"Rows containing at least one '?': {(df.astype(str) == '?').any(axis=1).sum()}")
    return question_counts


def step_4_handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4 — Delete rows with '?'; reject imputation for fairness reasons."""
    print("\n" + "=" * 72)
    print("STEP 4 — Handle missing values (delete vs impute)")
    print("=" * 72)
    n_before = len(df)
    mask_missing = (df.astype(str) == "?").any(axis=1)
    n_dropped = int(mask_missing.sum())
    cleaned = df.loc[~mask_missing].copy()
    n_after = len(cleaned)
    pct_lost = 100.0 * n_dropped / n_before

    print(
        "Decision: DELETE rows containing '?'.\n"
        "Alternative considered: IMPUTE (e.g. mode of workclass).\n"
        f"Justification: Deletion loses {n_dropped:,} of {n_before:,} records "
        f"({pct_lost:.2f}%, under 6%: {n_before:,} → {n_after:,}), which is "
        "small. Imputation was rejected because inventing values risks "
        "distorting exactly the under-recorded groups a fairness audit exists "
        "to protect (cf. Pagano et al.).\n"
    )
    print(f"Shape after deletion: {cleaned.shape}")
    return cleaned


def step_5_drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5 — Drop the row-index column `x`."""
    print("\n" + "=" * 72)
    print("STEP 5 — Remove genuinely useless columns")
    print("=" * 72)
    print(
        "Decision: Drop `x`.\n"
        "Justification: `x` is only a row number — an identifier with no "
        "predictive or analytical value. Keeping it could confuse a model later.\n"
    )
    return df.drop(columns=["x"])


def step_6_check_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6 — Verify outcome and protected columns are consistent."""
    print("\n" + "=" * 72)
    print("STEP 6 — Check outcome and protected columns")
    print("=" * 72)
    print(
        "Justification: Fairness calculations depend on these columns being "
        "consistent, so verifying labels (and stripping stray spaces) is "
        "necessary preparation.\n"
    )
    for col in (OUTCOME, PROTECTED_ATTRIBUTE, "race"):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
        print(f"{col} value counts:")
        print(df[col].value_counts())
        print(f"{col} unique: {sorted(df[col].unique())}\n")
    return df


def step_7_confirm_final_state(df: pd.DataFrame) -> None:
    """Step 7 — Confirm shape and that no '?' remain."""
    print("=" * 72)
    print("STEP 7 — Confirm final state")
    print("=" * 72)
    print(
        "Justification: Verify cleaning did what was intended rather than "
        "assuming it — closing the loop.\n"
    )
    remaining_q = int((df.astype(str) == "?").sum().sum())
    print(f"Final df.shape: {df.shape}  (expected (46043, 9))")
    print(f"Remaining '?': {remaining_q}")
    assert remaining_q == 0, "Expected no '?' after cleaning"
    assert df.shape == (46043, 9), f"Expected (46043, 9), got {df.shape}"
    print("Checks passed.")


def step_8_note_deferred_work() -> None:
    """Step 8 — Note preparation deliberately deferred."""
    print("\n" + "=" * 72)
    print("STEP 8 — What we did NOT do (yet), and why")
    print("=" * 72)
    print(
        "Scaling numeric features and encoding categorical columns are "
        "deferred to the modelling stage. Current disparity / selection-rate "
        "measurement operates on raw categories; encoding is only required "
        "once a classifier is trained.\n"
    )


def main() -> pd.DataFrame:
    # Stop after Step 1 — inspect before any cleaning.
    return step_1_load_and_inspect(RAW_CSV)


if __name__ == "__main__":
    main()
