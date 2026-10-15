# 🔍 Calibrated RAG Verification: Hallucination Detection in Retrieval-Augmented Generation

> **Course:** Natural Language Processing (NLP) — 7th Semester  
> **Assignment:** Experiment 10: Implementation and Presentation of NLP Mini Project  
> **Student Name:** Junaid  
> **Batch / Roll:** B23  
> **Base Reference Paper:** *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models* (ACL 2024)  
> **Paper Link:** [ACL Anthology: 2024.acl-long.585](https://aclanthology.org/2024.acl-long.585/)

---

## 📌 1. Project Overview & Motivation

Retrieval-Augmented Generation (**RAG**) architectures reduce factual errors in Large Language Models (LLMs) by augmenting queries with relevant context documents. However, recent findings in **RAGTruth (ACL 2024)** reveal that even top-tier models (GPT-4, Llama-2-70B) continue to hallucinate in **18% to 42%** of generation turns due to knowledge conflicts, input saturation, and erroneous extrapolation.

In RAG, hallucinations manifest primarily in two forms:
1. **Factual Conflicts (Contradictions):** Direct factual discrepancies with information explicitly stated in retrieved context.
2. **Baseless Claims (Unsupported Information):** Introducing ungrounded entities, dates, or specifications absent from the reference text.

This project delivers an end-to-end **Calibrated Hallucination Verification Pipeline** that detects hallucinations, calibrates confidence scores via **Temperature Scaling**, and achieves reliable post-generation verification.

---

## 🏗️ 2. System Architecture

```mermaid
flowchart TD
    A[Query + Retrieved Context] --> B[RAG Response Generation]
    B --> C[NLP Preprocessing Pipeline]
    C --> D1[Method 1: ROUGE Token Overlap]
    C --> D2[Method 2: TF-IDF Cosine Semantic Similarity]
    C --> D3[Method 3: Cross-Encoder NLI Entailment]
    D1 --> E[Post-Hoc Temperature Scaling Calibration]
    D2 --> E
    D3 --> E
    E --> F[Weighted Ensemble Verification]
    F --> G{Confidence > Threshold?}
    G -->|Yes| H[Flagged: Hallucinated / Reject Response]
    G -->|No| I[Verified: Faithful / Accept Response]
```

---

## 📊 3. Dataset Construction (RAGTruth-Inspired)

A structured benchmark of **30 test examples** was constructed spanning three diverse NLP task modalities:

| Modality | Examples | Faithful (0) | Hallucinated (1) | Core Verification Challenge |
| :--- | :---: | :---: | :---: | :--- |
| **Question Answering (QA)** | 10 | 5 | 5 | Precise fact extraction, entity names, and historical dates |
| **News Summarization** | 10 | 5 | 5 | Multi-sentence synthesis, preservation of statistical values |
| **Data-to-Text Generation** | 10 | 5 | 5 | Complex structured JSON attributes (pricing, ratings, amenities) |
| **Total** | **30** | **15** | **15** | **Perfect 50/50 Class Balance** |

### Exploratory Data Analysis Output
![Exploratory Data Analysis](eda_plots.png)
*Figure 1: Label balance, task type proportions, hallucination categories, and text length distributions.*

---

## 🕵️ 4. Verification Methodology

### Method 1: ROUGE-1 / ROUGE-2 Token Overlap (Baseline)
- Evaluates lexical precision: the ratio of generated response tokens grounded in the retrieved context.
- **Strength:** Extremely fast, zero GPU overhead.
- **Weakness:** Blind to numerical replacements and logical negation when surrounding vocabulary remains identical.

### Method 2: TF-IDF Cosine Semantic Similarity
- Vectorizes contexts and generated responses into 3000-dimensional TF-IDF vectors using unigrams and bigrams with sublinear term-frequency scaling.
- Evaluates global topical coherence in vector space:
$$\text{Cosine Similarity} = \frac{\mathbf{v}_{\text{context}} \cdot \mathbf{v}_{\text{response}}}{\|\mathbf{v}_{\text{context}}\| \|\mathbf{v}_{\text{response}}\|}$$

### Method 3: Cross-Encoder Natural Language Inference (NLI)
- Analyzes premise-hypothesis pairs through cross-attention to infer semantic entailment probability ($P(\text{entailment})$).
- Direct sensitivity to polar negation and factual contradictions.

### Post-Hoc Confidence Calibration (Temperature Scaling)
Raw confidence probabilities $p$ often display overconfidence. We apply post-hoc Temperature Scaling:
$$p_{\text{cal}} = \sigma\left(\frac{\text{logit}(p)}{T}\right)$$
Calibration quality is evaluated via **Expected Calibration Error (ECE)**:
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

![Calibration Curves](calibration_curves.png)
*Figure 2: Reliability Diagrams before and after Temperature Scaling calibration.*

### Weighted Multi-Model Ensemble
Each architecture's strengths are combined through an optimal weighted voting scheme:
$$\text{Score}_{\text{Ensemble}} = 0.20 \times \text{Score}_{\text{ROUGE}} + 0.30 \times \text{Score}_{\text{TF-IDF}} + 0.50 \times \text{Score}_{\text{NLI}}$$

---

## 📈 5. Experimental Results & Performance

| Architecture | Precision | Recall | F1 Score | AUC-ROC | Temperature $T$ | ECE (Calibrated) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Method 1: ROUGE Overlap** | 0.7143 | 0.7333 | 0.7237 | 0.7822 | 1.5 | 0.1345 |
| **Method 2: TF-IDF Cosine** | 0.8000 | 0.8000 | 0.8000 | 0.8578 | 1.3 | 0.1120 |
| **Method 3: NLI Entailment** | **0.9333** | **0.9333** | **0.9333** | **0.9733** | 1.2 | **0.0712** |
| **⭐ Weighted Ensemble** | **0.9333** | **0.9333** | **0.9333** | **0.9822** | — | **Optimal** |

### Evaluation Dashboard
![Evaluation Dashboard](evaluation_dashboard.png)
*Figure 3: ROC curves, confusion matrices, metric comparisons, and confidence violin plots.*

### Threshold Sensitivity & Score Distributions
![Score Distributions](score_distributions.png)
*Figure 4: F1 curves across decision thresholds and score density separations.*

---

## 🔬 6. Key Findings & Discussion

1. **Semantic Entailment Outperforms Lexical Metrics:** NLI achieved an **F1 score of 0.933** compared to ROUGE's **0.724**. Factual alterations (e.g. date substitutions) cannot be detected purely by token overlap.
2. **Data-to-Text exhibits highest vulnerability:** Attribute-dense structured JSON representations create numerous independent opportunities for entity substitution.
3. **Calibration reduces overconfidence:** Temperature scaling reduced ECE across all models (NLI ECE dropped from 0.114 to 0.071), making probability outputs dependable for risk-critical deployment.
4. **Comparison to RAGTruth (ACL 2024):** Our results confirm the paper's central conclusion: lexical baselines are fundamentally insufficient for hallucination detection; deeper entailment and calibrated ensembling are required.

---

## 📁 7. Submission Deliverables Summary

| File | Description | Location | Status |
| :--- | :--- | :--- | :---: |
| **`Calibrated_RAG_Verification.ipynb`** | Complete, executable Jupyter Notebook (all 30 cells validated) | Repository Root | ✅ Committed & Pushed to `main` |
| **`Calibrated_RAG_Verification.pptx`** | 12-slide comprehensive presentation deck for project defense | Repository Root | ✅ Ready |
| **`Ref-Paper-2024.acl-long.585.pdf`** | ACL 2024 Base Reference Paper (*RAGTruth*) | Repository Root | ✅ Attached |
| **`Junaid_B23_EXP10.docx`** | Official Lab Manual Experiment 10 Report (with embedded figures) | `C:\Users\junai\Desktop\NLP_docx\` | ✅ Complete |
| **`Experiment_10_Report_Junaid_B23.docx`** | Project Documentation Report with all output screenshots | Project Directory | ✅ Complete |
| **Output Plots (`*.png`)** | High-resolution output figures (EDA, Calibration, Dashboard, Scores) | Project Directory | ✅ Generated |

---

## 🚀 8. Quick Start / How to Run

### Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk torch transformers sentence-transformers python-docx
```

### Run Jupyter Notebook
```bash
jupyter notebook Calibrated_RAG_Verification.ipynb
```

### Re-run Verification Pipeline & Generate Figures
```bash
python generate_plots.py
```

### Re-build Word Documentation
```bash
python create_exp10_doc.py
```
