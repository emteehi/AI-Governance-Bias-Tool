# AI-Governance-Bias-Tool
Decision-making system that assesses the results against the requirements of the EU Artificial Intelligence Act.

# Step 1
Create virtual environment
```bash
python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install streamlit fairlearn ucimlrepo pandas
```

# Explore the dataset (interactive)
```bash
.venv/bin/streamlit run app.py
```

Opens a Streamlit app focused on the fairness-critical columns (`gender`, `race`, `income`) with filters, selection-rate charts, and feature distributions.