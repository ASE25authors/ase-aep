
# Advancing Automated Ethical Profiling in SE: a Zero-Shot Evaluation of LLM Reasoning

**Replication Package for the Paper:**  
*Advancing Automated Ethical Profiling in SE: a Zero-Shot Evaluation of LLM Reasoning*

---

## Contents

This package provides all code, data, and instructions to fully reproduce the experimental results, tables, and figures presented in the paper. The evaluation pipeline covers prompt construction, batch LLM querying (API/offline), parsing, aggregation, and both quantitative and qualitative analysis.

### File List

| File                               | Description                                                                                       |
|------------------------------------|---------------------------------------------------------------------------------------------------|
| `ethic_questionnaire.pdf`          | Original human ethics questionnaire (source for scenarios)                                        |
| `scenarios.txt`                    | 30 declarative ethically-charged scenarios for LLM prompting                                      |
| `endpoints.txt`                    | List of all LLMs (16) with API endpoints and offline models, as used in the experiment            |
| `prompting.py`                     | Main script: prompts all LLMs, collects and saves their responses                                 |
| `raw_xlsx_parsing.py`              | Script to parse raw responses and compute TCR, BAR, and structure data for analysis               |
| `qualitative.py`                   | Analysis script: computes all qualitative results (clustering, topic modeling, wordcloud, etc.)   |
| `requirements.txt`                 | Python requirements for code execution                                                            |
| `llm_outputs_parsed.xlsx`          | Parsed and structured dataset with all intermediate metrics and response breakdown                |
| `experts.csv`                      | Expert ethicists' responses                                                                       |
| `LLMs Ethical Comparison.xlsx`     | Complete edited dataset with conditional formatting for visual feedback on thresholds             |
| `LDA_topic_labels.txt`             | Provides an illustrative, analyst-facing mapping from LDA topics to short human-readable labels.  |
| `\images`                          | Complete set of the produced images                                                               |
| `README.md`                        | This file                                                                                         |

---

## Environment Setup

**Recommended Python version:** 3.10+  
**Required packages:**  
- `pandas`
- `openpyxl`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `gensim`
- `tqdm`
- `wordcloud`
- `transformers`
- `requests`
- `psutil`
- `numpy`

Install with:

```bash
pip install -r requirements.txt
```

Or manually, for core functions:

```bash
pip install pandas openpyxl matplotlib seaborn scikit-learn gensim tqdm wordcloud transformers requests psutil numpy
```

---

## Pipeline Overview

### 1. Prompting LLMs (API & Offline)

- **Configure API keys in `endpoints.txt`.**
- Run:

  ```bash
  python prompting.py
  ```
  This will:
  - Read `scenarios.txt` and `endpoints.txt`
  - Send prompts to each LLM (API and offline)
  - Save responses as `llm_raw_outputs.xlsx`

### 2. Parsing and Metric Extraction

- Run:

  ```bash
  python raw_xlsx_parsing.py
  ```
  This will:
  - Read `llm_raw_outputs.xlsx`
  - Parse each response into: Ethical Theory, Morally Acceptable (yes/no), Explanation
  - Compute TCR (Theory Consistency Rate) and BAR (Binary Agreement Rate)
  - Write all results to `llm_outputs_parsed.xlsx`

### 3. Qualitative and Quantitative Analysis

- Run:

  ```bash
  python qualitative.py
  ```
  This will:
  - Analyze the explanations for all LLMs/scenarios
  - Generate figures: LDA topic modeling, PCA/t-SNE semantic clustering, word counts, word clouds, similarity heatmaps
  - Save all results in `.png` files as referenced in the paper

---

## Data and Format Details

- **`scenarios.txt`:** One scenario per line, in declarative form.
- **`llm_raw_outputs.xlsx`:** Each cell (except first column) contains the unparsed response:  
- **`llm_outputs_parsed.xlsx`:** Multi-column, fully structured for statistical and qualitative analysis.

---

## Notes and Recommendations

- **For full API-based runs:** Insert your own API keys into `endpoints.txt`. Some endpoints require credentials.
- **For offline LLMs:** Make sure required HuggingFace models are downloaded and available on your machine in the right path.

---

## Reproducibility Checklist

- [x] All scenarios and code are included
- [x] Data for all LLMs as reported in the paper are present
- [x] All scripts can be executed step-by-step from raw to figure
- [X] Expert ethicists' answers available for benchmarking

For any issues, please contact the corresponding author.

---

## Citation

If you use this code or data, please cite the paper:

> [Full citation info from the paper]

---
