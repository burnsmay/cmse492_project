<<<<<<< HEAD
# Fake News Detection – CMSE 492 Project

This project aims to build a Fake News Detection system using machine learning techniques to automatically classify online news articles as real or fake. Using the Fake.csv and True.csv datasets, I will preprocess data including the title, body text, subject, and publish date of each news article to prepare it for analysis. The project will explore both classical machine learning models, such as Logistic Regression and Support Vector Machines, and deep learning approaches to evaluate which methods perform best at detecting misinformation.



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
   git clone https://github.com/burnsmay/cmse492_project.git
   cd cmse492_project
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt


## How to Run
### 1. Data Preprocessing
Run scripts in `src/preprocessing/` to clean and prepare the datasets.  
This will generate the processed CSV files in `data/processed/`.
=======
# cmse492_project
>>>>>>> ec823b5 (Update notebook and project folders)
