"""
Interactive Streamlit explorer for the Adult / Census Income dataset.

Focus: protected attributes (gender, race) and outcome (income) for the
fairness / EU AI Act audit workflow.
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV = DATA_DIR / "adult.csv"
CLEAN_CSV = DATA_DIR / "adult_clean.csv"

PROTECTED_PRIMARY = "gender"
PROTECTED_SECONDARY = "race"
OUTCOME = "income"
POSITIVE_OUTCOME = ">50K"

COLUMN_ROLES = {
    "x": "Row identifier (no analytical value)",
    "age": "Numeric feature — age in years",
    "workclass": "Categorical feature — type of employer",
    "education": "Categorical feature — highest education level",
    "marital-status": "Categorical feature — marital status",
    "relationship": "Categorical feature — household relationship",
    "race": "Protected attribute (secondary audit)",
    "gender": "Protected attribute (primary audit)",
    "hours-per-week": "Numeric feature — hours worked per week",
    "income": "Outcome — income band (<=50K / >50K)",
}


@st.cache_data
def load_data(source: str) -> pd.DataFrame:
    path = CLEAN_CSV if source == "Cleaned" else RAW_CSV
    df = pd.read_csv(path)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    with st.sidebar:
        st.header("Filters")
        st.caption("Narrow the view before exploring charts.")

        age_min, age_max = int(df["age"].min()), int(df["age"].max())
        age_range = st.slider("Age", age_min, age_max, (age_min, age_max))
        filtered = filtered[filtered["age"].between(*age_range)]

        hours_min, hours_max = int(df["hours-per-week"].min()), int(df["hours-per-week"].max())
        hours_range = st.slider(
            "Hours per week", hours_min, hours_max, (hours_min, hours_max)
        )
        filtered = filtered[filtered["hours-per-week"].between(*hours_range)]

        for col in ("gender", "race", "income", "education", "workclass"):
            if col not in filtered.columns:
                continue
            options = sorted(df[col].dropna().unique().tolist())
            selected = st.multiselect(col.replace("-", " ").title(), options, default=options)
            if selected:
                filtered = filtered[filtered[col].isin(selected)]
            else:
                filtered = filtered.iloc[0:0]

    return filtered


def selection_rate_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Share of each group with income >50K (positive outcome rate)."""
    rates = (
        df.groupby(group_col, observed=True)[OUTCOME]
        .agg(
            n="count",
            n_high=lambda s: (s == POSITIVE_OUTCOME).sum(),
        )
        .reset_index()
    )
    rates["selection_rate"] = rates["n_high"] / rates["n"]
    rates["selection_rate_pct"] = (rates["selection_rate"] * 100).round(1)
    return rates.sort_values("selection_rate", ascending=False)


def bar_counts(df: pd.DataFrame, col: str, title: str) -> alt.Chart:
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y(f"{col}:N", sort="-x", title=col.replace("-", " ").title()),
            tooltip=[col, "count"],
            color=alt.value("#2E6F9E"),
        )
        .properties(title=title, height=max(180, 28 * len(counts)))
    )


def grouped_outcome_chart(df: pd.DataFrame, group_col: str, title: str) -> alt.Chart:
    counts = (
        df.groupby([group_col, OUTCOME], observed=True)
        .size()
        .reset_index(name="count")
    )
    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X(f"{group_col}:N", title=group_col.replace("-", " ").title()),
            y=alt.Y("count:Q", title="Count", stack="normalize"),
            color=alt.Color(
                f"{OUTCOME}:N",
                title="Income",
                scale=alt.Scale(range=["#8FA6B8", "#1B4F72"]),
            ),
            tooltip=[group_col, OUTCOME, "count"],
        )
        .properties(title=title, height=320)
    )


def selection_rate_chart(rates: pd.DataFrame, group_col: str, title: str) -> alt.Chart:
    return (
        alt.Chart(rates)
        .mark_bar()
        .encode(
            x=alt.X("selection_rate_pct:Q", title="% with income >50K"),
            y=alt.Y(f"{group_col}:N", sort="-x", title=group_col.replace("-", " ").title()),
            tooltip=[
                group_col,
                alt.Tooltip("n:Q", title="Group size"),
                alt.Tooltip("n_high:Q", title=">50K count"),
                alt.Tooltip("selection_rate_pct:Q", title="Selection rate %"),
            ],
            color=alt.value("#C0392B"),
        )
        .properties(title=title, height=max(180, 36 * len(rates)))
    )


def numeric_hist(df: pd.DataFrame, col: str, title: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=40), title=col.replace("-", " ").title()),
            y=alt.Y("count()", title="Count"),
            color=alt.Color(
                f"{OUTCOME}:N",
                title="Income",
                scale=alt.Scale(range=["#8FA6B8", "#1B4F72"]),
            ),
            tooltip=["count()", OUTCOME],
        )
        .properties(title=title, height=280)
    )


def main() -> None:
    st.set_page_config(
        page_title="Census Income Explorer",
        page_icon="📊",
        layout="wide",
    )

    st.title("Adult / Census Income — interactive explorer")
    st.markdown(
        "Explore the dataset used for the fairness audit. "
        f"**Primary protected attribute:** `{PROTECTED_PRIMARY}` · "
        f"**Outcome:** `{OUTCOME}` · "
        f"**Secondary protected attribute:** `{PROTECTED_SECONDARY}`."
    )

    with st.sidebar:
        source = st.radio("Dataset version", ["Raw", "Cleaned"], index=1)
        st.divider()

    df = load_data(source)
    filtered = apply_filters(df)

    # --- KPI strip ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (filtered)", f"{len(filtered):,}")
    c2.metric("Columns", len(filtered.columns))
    if len(filtered):
        high_share = (filtered[OUTCOME] == POSITIVE_OUTCOME).mean() * 100
        female_share = (filtered[PROTECTED_PRIMARY] == "Female").mean() * 100
    else:
        high_share = female_share = 0.0
    c3.metric("% with income >50K", f"{high_share:.1f}%")
    c4.metric("% Female", f"{female_share:.1f}%")

    tab_overview, tab_fairness, tab_features, tab_table = st.tabs(
        ["Overview", "Fairness lens", "Features", "Data table"]
    )

    with tab_overview:
        st.subheader("Column roles")
        roles = pd.DataFrame(
            {
                "column": list(filtered.columns),
                "role": [COLUMN_ROLES.get(c, "(see dataset docs)") for c in filtered.columns],
            }
        )
        st.dataframe(roles, use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            if len(filtered):
                st.altair_chart(
                    bar_counts(filtered, OUTCOME, "Outcome distribution"),
                    use_container_width=True,
                )
        with right:
            if len(filtered):
                st.altair_chart(
                    bar_counts(filtered, PROTECTED_PRIMARY, "Gender distribution"),
                    use_container_width=True,
                )

        if len(filtered) and PROTECTED_SECONDARY in filtered.columns:
            st.altair_chart(
                bar_counts(filtered, PROTECTED_SECONDARY, "Race distribution"),
                use_container_width=True,
            )

    with tab_fairness:
        st.subheader("Outcome rates by protected group")
        st.caption(
            "Selection rate = share of the group with income `>50K`. "
            "Large gaps between groups are the starting point for the bias audit."
        )

        if not len(filtered):
            st.warning("No rows match the current filters.")
        else:
            g_col, r_col = st.columns(2)

            with g_col:
                gender_rates = selection_rate_table(filtered, PROTECTED_PRIMARY)
                st.altair_chart(
                    selection_rate_chart(
                        gender_rates,
                        PROTECTED_PRIMARY,
                        "Selection rate by gender",
                    ),
                    use_container_width=True,
                )
                st.dataframe(
                    gender_rates.rename(
                        columns={
                            "n": "group size",
                            "n_high": ">50K",
                            "selection_rate_pct": "rate %",
                        }
                    )[[PROTECTED_PRIMARY, "group size", ">50K", "rate %"]],
                    use_container_width=True,
                    hide_index=True,
                )
                if len(gender_rates) >= 2:
                    gap = gender_rates["selection_rate"].max() - gender_rates["selection_rate"].min()
                    st.info(f"Gender selection-rate gap (max − min): **{gap * 100:.1f} percentage points**")

            with r_col:
                if PROTECTED_SECONDARY in filtered.columns:
                    race_rates = selection_rate_table(filtered, PROTECTED_SECONDARY)
                    st.altair_chart(
                        selection_rate_chart(
                            race_rates,
                            PROTECTED_SECONDARY,
                            "Selection rate by race",
                        ),
                        use_container_width=True,
                    )
                    st.dataframe(
                        race_rates.rename(
                            columns={
                                "n": "group size",
                                "n_high": ">50K",
                                "selection_rate_pct": "rate %",
                            }
                        )[[PROTECTED_SECONDARY, "group size", ">50K", "rate %"]],
                        use_container_width=True,
                        hide_index=True,
                    )

            st.divider()
            st.markdown("#### Stacked outcome mix (normalized)")
            s1, s2 = st.columns(2)
            with s1:
                st.altair_chart(
                    grouped_outcome_chart(
                        filtered, PROTECTED_PRIMARY, "Income mix by gender"
                    ),
                    use_container_width=True,
                )
            with s2:
                if PROTECTED_SECONDARY in filtered.columns:
                    st.altair_chart(
                        grouped_outcome_chart(
                            filtered, PROTECTED_SECONDARY, "Income mix by race"
                        ),
                        use_container_width=True,
                    )

    with tab_features:
        st.subheader("Feature distributions")
        if not len(filtered):
            st.warning("No rows match the current filters.")
        else:
            n1, n2 = st.columns(2)
            with n1:
                st.altair_chart(
                    numeric_hist(filtered, "age", "Age by income"),
                    use_container_width=True,
                )
            with n2:
                st.altair_chart(
                    numeric_hist(filtered, "hours-per-week", "Hours/week by income"),
                    use_container_width=True,
                )

            cat_col = st.selectbox(
                "Categorical feature",
                [
                    c
                    for c in (
                        "education",
                        "workclass",
                        "marital-status",
                        "relationship",
                    )
                    if c in filtered.columns
                ],
            )
            st.altair_chart(
                grouped_outcome_chart(
                    filtered, cat_col, f"Income mix by {cat_col}"
                ),
                use_container_width=True,
            )
            st.altair_chart(
                selection_rate_chart(
                    selection_rate_table(filtered, cat_col),
                    cat_col,
                    f"Selection rate by {cat_col}",
                ),
                use_container_width=True,
            )

    with tab_table:
        st.subheader("Browse rows")
        st.caption(f"Showing up to 500 of {len(filtered):,} filtered rows.")
        display_cols = [c for c in filtered.columns if c != "x"]
        st.dataframe(
            filtered[display_cols].head(500),
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered CSV",
            data=csv_bytes,
            file_name="adult_filtered.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
