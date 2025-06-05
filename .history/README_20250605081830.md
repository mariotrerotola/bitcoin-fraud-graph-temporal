# ChainAbuse + Elliptic++: Data Extraction, Feature Engineering, and Explainability Framework

## 1. Project Overview

This repository assembles three complementary data sources—**ChainAbuse**, **Elliptic++**, and **Blockchain.com**—to construct graph‑temporal features for Bitcoin wallet classification.  We benchmark three model families (Random Forest, MLP, SVM) and evaluate their explainability through **SHAP**, **CIU**, and **DEXiRE**.

> **Key objective** Quantitatively compare classical ML, neural distillation, and post‑hoc explainability in the presence of label noise and feature ablation.

---

## 2. Data Sources

| Source                  | Link                                                                                         | Notes                                                                       |
| ----------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| ChainAbuse reports      | [https://www.chainabuse.com](https://www.chainabuse.com)                                     | Scraped via the `chainabuse_fetcher` notebooks.                             |
| Elliptic++ dataset      | [https://github.com/git-disl/EllipticPlusPlus](https://github.com/git-disl/EllipticPlusPlus) | Augments the original Elliptic dataset with additional illicit labels.      |
| Blockchain Explorer API | [https://www.blockchain.com/explorer/api](https://www.blockchain.com/explorer/api)           | Used (rate‑limited) to fetch raw transaction graphs for individual wallets. |

---

## 3. Recommended Environment

| Requirement | Version                                                 |
| ----------- | ------------------------------------------------------- |
| **Python**  | **3.10.16**  (DEXiRE is not yet compatible with ≥ 3.11) |

Create a fresh virtual environment and install dependencies:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # from the project root
```

> The `requirements.txt` file pins all libraries for full reproducibility.

After installing **Playwright**, run `playwright install` to download the headless browsers.

---

## 4. Repository Structure

```
├── chainabuse_fetcher/
│   ├── chainabuse_fetcher.ipynb   # Scraping + JSON/CSV export
│   └── clean_data.ipynb          # Pre‑processing & deduplication
│
├── wallet_fetcher.ipynb          # Wallet‑level transaction extraction via Blockchain API
├── extract_metrics.ipynb         # Graph‑temporal feature engineering
├── dataset_assembly.ipynb        # Construction of Case 1, 2, 3 + feature cleaning
├── data_exploration.ipynb        # Descriptive statistics for each case
│
├── confident_learning/
│   └── confident_learning.ipynb  # Label‑noise detection with CleanLab
│
├── ml_and_explainability/
│   ├── rf/
│   │   └── random_forest_with_explainability.ipynb
│   ├── rf_mlp_distill_dexire.ipynb
│   └── svm/
│       └── svm_gridsearch.ipynb
│
└── ablation_study/
    └── ablation_study.ipynb      # Feature ablation (1‑ and 2‑feature drops)
```

---

## 5. Notebook Highlights

### 5.1 `random_forest_with_explainability.ipynb`

* **Hyper‑parameter search** via GridSearchCV.
* Comprehensive performance report.
* Global and local explanations through **SHAP** and **Contrastive Influence Utility (CIU)**.

### 5.2 `rf_mlp_distill_dexire.ipynb`

* Knowledge‑distills the best Random Forest into an MLP surrogate.
* Applies **DEXiRE** for symbolic rule extraction and fidelity analysis.

### 5.3 `svm_gridsearch.ipynb`

* Exhaustive grid search over kernels and C/γ parameters.

### 5.4 `ablation_study.ipynb`

* Iteratively removes one or two features, retrains a Random Forest, and measures Δ‑accuracy.

---

## 6. Reproducing the Pipeline

1. **Scrape ChainAbuse** reports: run `chainabuse_fetcher.ipynb` (headless Chromium).
2. **Clean & unify** all reports via `clean_data.ipynb`.
3. **Wallet‑level transactions**: execute `wallet_fetcher.ipynb` (API keys not required, but rate‑limited).
4. **Feature extraction**: run `extract_metrics.ipynb` to compute in/out degrees, net balances, temporal intervals, etc.
5. **Assemble cases**: launch `dataset_assembly.ipynb` to merge ChainAbuse, Elliptic++, and Blockchain features.
6. **Exploratory analysis**: open `data_exploration.ipynb` for descriptive stats and correlation matrices.
7. **Detect label noise**: run `confident_learning.ipynb` (CleanLab) and revise labels when appropriate.
8. **Model training and explainability**: follow the notebooks in `ml_and_explainability/`.
9. **Ablation study**: quantify feature importance via accuracy drop.
