# 🏨 Hotel Reviews Analysis & Topic Modeling (R)

This project performs sentiment-based text mining and topic modeling on hotel review data using R. The goal is to extract insights from positive and negative customer reviews through word frequency analysis and **Latent Dirichlet Allocation (LDA)** topic modeling.

---

## 🧠 Technologies & Libraries Used

- **Language:** R  
- **Main packages:** `dplyr`, `ggplot2`, `tm`, `cld2`, `wordcloud`, `topicmodels`, `LDAvis`, `readr`, `stringr`, `servr`

---

## 📦 Project Workflow

### 1. Data Ingestion & Preprocessing
- Data is loaded from `HotelsData.csv`
- Language detection is performed using the `cld2` package
- Reviews are categorized into:
  - **Positive reviews**: scores 4 and 5
  - **Negative reviews**: scores 1 and 2
  - **Neutral reviews**: score 3

### 2. Sampling and Language Filtering
- 1,000 English-language reviews are randomly sampled from each sentiment category (positive & negative)
- Language distribution is summarized for the entire dataset

### 3. Text Preprocessing
Applied separately for positive and negative samples:
- Conversion to corpus using `tm`
- Tokenization, lowercasing, stop word removal, punctuation and number removal
- Creation of Document-Term Matrix (DTM)
- Removal of empty documents

### 4. Word Frequency Analysis
- Term frequency is calculated and sorted
- **Word clouds** are generated for both positive and negative reviews to visualize the most frequent terms

### 5. Topic Modeling (LDA)
- Latent Dirichlet Allocation (LDA) is applied to both positive and negative DTMs
- Number of topics (`k`) set to 3
- Gibbs sampling is used with 1,000 iterations and a fixed seed for reproducibility
- **Interactive topic visualizations** are created with `LDAvis`

---

## 📊 Key Outputs

- Language breakdown of reviews
- Summary table of sentiment distribution (positive, neutral, negative)
- Word clouds for both sentiment groups
- LDA topic models for positive and negative reviews
- Web-based visualizations of topics (via `serVis()`)
