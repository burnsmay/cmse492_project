# Fake News Detection – CMSE 492 Project

This project aims to build a Fake News Detection system using machine learning techniques to automatically classify online news articles as real or fake. Using the Fake.csv and True.csv datasets, I will preprocess data including the title, body text, subject, and publish date of each news article to prepare it for analysis. The project will explore both classical machine learning models, such as Logistic Regression and Support Vector Machines, and deep learning approaches to evaluate which methods perform best at detecting misinformation.

## Dataset
The dataset used for this project consists of labeled news articles collected from a combination of verified and unverified online media sources. The **Fake and Real News Dataset** was originally compiled to support research in misinformation detection and natural language processing. It contains **44,898 news articles** published by both credible organizations (e.g., Reuters, BBC, The Guardian) and sources known for disseminating fabricated or misleading stories. Each article includes the full text and a binary label indicating whether it is real or fake, based on verification against fact-checking databases and source credibility assessments.  

The dataset’s purpose is to provide a reliable foundation for developing machine learning models capable of identifying deceptive content patterns in digital journalism.

### Dataset Characteristics
- **Number of samples (rows):** 44,898 articles  
- **Number of features (columns):** 4  
  - `title` — the headline of the article  
  - `text` — full body text of the article  
  - `subject` — topic category (e.g., politics, technology)  
  - `label` — binary indicator of news veracity: `1 = REAL`, `0 = FAKE`  
- **Data types:** Primarily textual, with one categorical target variable  

### Data Quality
#### Missing Values
- Missing values are minimal: less than 0.2% of rows have missing `title` or `text`.  
- These rows were removed to avoid introducing noise during model training.  
- The remaining dataset is complete and suitable for text analysis.

#### Class Balance
- Real news articles: 21,417  
- Fake news articles: 23,481  
- The dataset is approximately balanced. Stratified splitting is used during training/testing to maintain this balance.


## Directory Structure
```
cmse492_project/
├── README.md # Project overview and documentation
├── .gitignore # Files and folders ignored by git
├── data/
│ ├── raw/ # Original datasets (unmodified)
│ └── processed/ # Cleaned and preprocessed datasets
├── notebooks/
│ └── exploratory/ # Jupyter notebooks for EDA and experiments
│ └── results/ # Notebook summarizing final model evaluation
├── src/
│ ├── preprocessing/ # Scripts for data cleaning and text preprocessing
│ ├── models/ # Model training and tuning scripts
│ └── evaluation/ # Scripts for model evaluation and visualization
├── figures/ # Generated plots and figures
├── docs/ # Documentation or project report
└── requirements.txt # List of dependencies
```

## Setup Instructions
**Clone the repository:**
```bash
   git clone https://github.com/burnsmay/cmse492_project.git
   cd cmse492_project
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
```

## How to Run
### 1. Data Preprocessing
Run scripts in `src/preprocessing/` to clean and prepare the datasets.  
This will generate the processed CSV files in `data/processed/`.

### 2. Training Models
Run the training scripts in `src/models/`:

- `logistic_regression.py`
- `svm_model.py`
- `bilstm_model.py`
These scripts save the trained models and vectorizers in the `models/` folder.

### 3. Evaluating Models
Use the scripts in `src/evaluation/` to generate:

- Metrics (accuracy, precision, recall, F1)
- Confusion matrices
- ROC curves
- Feature importance visualizations

### 4. Viewing Results
Open the **Results notebook**: notebooks/results/Final_Results.ipynb
This notebook summarizes:

- Model performance metrics (accuracy, precision, recall, F1)
- Training and inference times
- Feature importance visualizations
- Confusion matrices and ROC curves for each model
- Final selection of the best-performing model (Linear SVM)


