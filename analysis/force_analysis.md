# Force Analysis – `analyze_forces`

This file explains how the `analyze_forces` function works in detail.

---

## ⚙️ Purpose

This function computes the net ionic Coulomb force acting on a permeating ion from nearby ions in a given frame, and compares that force to the ion's motion and (optionally) the total force computed via OpenMM.

---

## 🧮 Key Steps

1. **Find permeating ion’s position at frame `t`**
2. **Loop through all other ions in the same frame**
3. **Compute Coulomb force if they are within a cutoff**
4. **Sum all forces to get net `ionic_force`**
5. **Break the force into radial and axial (z) components**
6. **Compare to motion vector**
7. **Store per-ion force contributions for downstream analysis**
8. **Optionally compare with `total_force` from OpenMM**

---

## 📐 Mathematical Basis

### Coulomb Force:

\[
\vec{F} = \frac{k \cdot q_1 q_2}{r^2} \cdot \hat{r}
\]

- \( k = 332 \) (in MD-friendly units: kcal·Å/(mol·e²))
- \( \hat{r} = \frac{\vec{r}_{12}}{|\vec{r}_{12}|} \)
- \( r \): Distance between two ions

---

## 🧠 Cosine Similarity

To assess directionality between vectors:

\[
\cos(\theta) = \frac{\vec{F} \cdot \vec{v}}{|\vec{F}||\vec{v}|}
\]

- \( \vec{F} \): Ionic force
- \( \vec{v} \): Motion vector (from `frame` to `frame+1`)
- Values close to +1 indicate alignment; -1 means opposition

---

## 🔁 Output Fields

```json
{
  "frame": 5003,
  "ionic_force": [1.23, -0.45, 2.78],
  "ionic_force_magnitude": 3.12,
  "motion_vector": [...],
  "cosine_ionic_motion": 0.94,
  "radial_force": 1.5,
  "axial_force": 2.78,
  "contributions": [
    {
      "ion": 1344,
      "force": [0.12, -0.03, 0.55],
      "cosine_with_motion": 0.82,
      ...
    }
  ]
}
