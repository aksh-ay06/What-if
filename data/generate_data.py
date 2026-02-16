"""
Generate a realistic synthetic lifestyle-outcome dataset with known causal relationships.

Causal structure (ground truth):
  sleep   → energy (+), mood (+), productivity (+)
  exercise → energy (+), mood (+), stress (-)
  diet_quality → energy (+), mood (+)
  screen_time  → sleep (-), stress (+), mood (-)
  caffeine     → energy (short-term +), sleep (-)
  stress       → mood (-), productivity (-), sleep (-)

Confounders injected so that naïve correlations ≠ causal effects.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 2000

# --- Exogenous factors (unobserved confounders) ---
age = np.random.normal(35, 10, N).clip(18, 65)
work_hours = np.random.normal(8, 2, N).clip(4, 14)

# --- Treatment variables (lifestyle habits) ---
# screen_time influenced by work_hours
screen_time = (2 + 0.4 * work_hours + np.random.normal(0, 1.5, N)).clip(1, 16)

# caffeine influenced by work_hours
caffeine = (1 + 0.3 * work_hours + np.random.normal(0, 1, N)).clip(0, 8)

# exercise influenced by age (older → less), work_hours (more work → less exercise)
exercise = (5 - 0.03 * age - 0.2 * work_hours + np.random.normal(0, 1.5, N)).clip(0, 14)

# Sleep depends on stress, but stress also depends on sleep — to break this
# simultaneity we use a preliminary "base_stress" estimate (without sleep) just
# to generate plausible sleep values. The final stress variable below is then
# computed using the generated sleep, giving us a proper causal ordering.
base_stress = (3 + 0.3 * work_hours - 0.15 * exercise + 0.1 * screen_time + np.random.normal(0, 1.5, N)).clip(1, 10)
sleep = (8 - 0.15 * screen_time - 0.2 * caffeine - 0.1 * base_stress + np.random.normal(0, 0.8, N)).clip(3, 12)

# diet_quality (1-10 scale)
diet_quality = (5 + 0.3 * exercise - 0.1 * work_hours + np.random.normal(0, 1.5, N)).clip(1, 10)

# --- Stress (outcome of exercise, screen_time; mediator) ---
stress = (3 + 0.3 * work_hours - 0.25 * exercise + 0.15 * screen_time
          - 0.1 * sleep + np.random.normal(0, 1.2, N)).clip(1, 10)

# --- Outcome variables ---
energy = (3 + 0.5 * sleep + 0.3 * exercise + 0.2 * diet_quality
          + 0.15 * caffeine - 0.1 * stress + np.random.normal(0, 1, N)).clip(1, 10)

mood = (3 + 0.35 * sleep + 0.3 * exercise + 0.2 * diet_quality
        - 0.25 * stress - 0.1 * screen_time + np.random.normal(0, 1, N)).clip(1, 10)

productivity = (2 + 0.35 * sleep + 0.2 * exercise + 0.15 * diet_quality
                + 0.1 * energy - 0.3 * stress - 0.05 * screen_time
                + np.random.normal(0, 1, N)).clip(1, 10)

# --- Assemble DataFrame ---
df = pd.DataFrame({
    "sleep": np.round(sleep, 1),
    "exercise": np.round(exercise, 1),
    "diet_quality": np.round(diet_quality, 1),
    "screen_time": np.round(screen_time, 1),
    "caffeine": np.round(caffeine, 1),
    "stress": np.round(stress, 1),
    "energy": np.round(energy, 1),
    "mood": np.round(mood, 1),
    "productivity": np.round(productivity, 1),
    # keep confounders for analysis, but they won't be user-facing
    "age": np.round(age, 0).astype(int),
    "work_hours": np.round(work_hours, 1),
})

out = Path(__file__).parent / "lifestyle_data.csv"
df.to_csv(out, index=False)
print(f"✓ Generated {len(df)} rows → {out}")
print(df.describe().round(2))
