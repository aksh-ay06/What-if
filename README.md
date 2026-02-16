# 🔬 Personal What-If: Causal Lifestyle Explorer

A lightweight Streamlit app where you input your lifestyle habits and explore causal what-if scenarios — e.g., *"What if I slept 1 hour more?"*

Uses a **causal DAG** with **DoWhy + EconML** to distinguish correlation from causation and provides trustworthy estimates backed by refutation tests.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit)
![DoWhy](https://img.shields.io/badge/DoWhy-Causal_Inference-green)

---

## Features

| Feature | Description |
|---------|-------------|
| **Lifestyle Input Form** | Sliders for sleep, exercise, diet, screen time, caffeine |
| **What-If Interventions** | Pick a habit to change, see estimated causal effect on energy, mood, productivity |
| **Causal DAG Visualization** | Interactive Plotly graph showing causal relationships |
| **Trustworthiness Indicators** | DoWhy refutation tests (placebo, random common cause, data subset) |
| **Personalized Estimates** | EconML's LinearDML for heterogeneous treatment effects based on your profile |

---

## Quick Start

```bash
# Clone the project
git clone <your-repo-url>
cd personal-what-if

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
personal-what-if/
├── app.py                   # Main Streamlit app
├── causal_model.py          # DoWhy/EconML model setup & inference
├── causal_graph.py          # DAG definition & Plotly visualization
├── utils.py                 # Helpers for formatting & charts
├── data/
│   ├── lifestyle_data.csv   # Synthetic dataset (2000 rows)
│   └── generate_data.py     # Script to regenerate data
├── requirements.txt
└── README.md
```

---

## Technical Approach

### Causal DAG
A domain-knowledge-based directed acyclic graph encodes causal relationships:
- **sleep → energy, mood, productivity**
- **exercise → energy, mood, stress**
- **screen_time → sleep, stress, mood**
- **caffeine → energy, sleep**
- Confounders: work_hours, age

### Causal Inference
- **DoWhy**: Defines the causal model, identifies estimands via the backdoor criterion, estimates ATE using linear regression
- **EconML (LinearDML)**: Heterogeneous treatment effects for personalised what-if estimates based on user profile

### Refutation Tests
Three DoWhy built-in checks validate estimate robustness:
1. **Placebo Treatment** — replaces treatment with random noise; effect should vanish
2. **Random Common Cause** — adds a random confounder; effect should remain stable
3. **Data Subset** — uses 80% of data; effect should remain stable

---

## Why This Project Stands Out

- **Causal inference** is underrepresented in portfolios — differentiates from typical ML projects
- **Interactive & relatable** — everyone understands lifestyle habits
- **Trustworthiness checks** demonstrate rigorous ML thinking beyond `model.fit()`
- **Zero ongoing cost** on Streamlit Cloud free tier
- **Strong interview talking point**: DAGs, confounders, treatment effects, refutation

---

## Deployment

Deploy for free on [Streamlit Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Point to `app.py` and deploy

---

## Key Libraries

| Library | Purpose |
|---------|---------|
| `streamlit` | Frontend UI |
| `dowhy` | Causal model definition & effect estimation |
| `econml` | Heterogeneous treatment effect estimation |
| `networkx` | DAG construction |
| `plotly` | Interactive charts & DAG visualization |
| `pandas` / `numpy` | Data handling |
| `scikit-learn` | ML models used inside EconML |

---

## License

MIT
