\# 🛡️ FraudLens AI — Credit Card Fraud Detection



An end-to-end machine learning project that detects fraudulent credit card transactions, complete with exploratory data analysis, model comparison, explainability (SHAP), and an interactive Streamlit web app.



\## 🔗 Live Demo

\[FraudLens AI — Live App](#) <!-- Add your Streamlit Cloud link here once deployed -->



\## 📌 Overview



Credit card fraud detection is a classic imbalanced classification problem — fraudulent transactions make up just \*\*0.17%\*\* of all transactions in this dataset. This project walks through the full pipeline: from raw data to a deployed, explainable ML application.



\## 📸 Screenshots



\### Dataset Overview

!\[Dataset Overview](screenshots/dataset\_overview.png)



\### Transaction Analyzer — Fraud Detected

!\[Transaction Analyzer - Fraud](screenshots/transaction\_analyzer\_fraud.png)



\### Transaction Analyzer — Genuine Transaction

!\[Transaction Analyzer - Genuine](screenshots/transaction\_analyzer\_genuine.png)



\### Model Performance

!\[Model Performance](screenshots/model\_performance.png)



\## 📊 Dataset



\- \*\*Source:\*\* \[Kaggle — Credit Card Fraud Detection (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

\- \*\*Size:\*\* 284,807 transactions, 492 fraudulent (0.17%)

\- \*\*Features:\*\* `Time`, `Amount`, and 28 anonymized PCA components (`V1`–`V28`)

\- \*\*Target:\*\* `Class` (0 = Genuine, 1 = Fraud)



\## 🧠 Project Workflow



| Notebook | Description |

|---|---|

| `01\_eda\_preprocessing.ipynb` | Exploratory data analysis, feature scaling, train/test split, SMOTE for class imbalance |

| `02\_modeling.ipynb` | Trained and compared Logistic Regression, Random Forest, and XGBoost |

| `03\_evaluation.ipynb` | Confusion matrix, ROC curve, Precision-Recall curve for the final model |

| `04\_shap.ipynb` | Model explainability using SHAP — global feature importance and individual prediction explanations |



\## 🏆 Model Performance (Random Forest — Final Model)



| Metric (Fraud Class) | Score |

|---|---|

| Precision | 0.82 |

| Recall | 0.82 |

| F1-Score | 0.82 |

| ROC-AUC | 0.9688 |

| Average Precision (PR-AUC) | 0.8678 |



Random Forest was selected as the final model after comparing against Logistic Regression and XGBoost, offering the best balance between precision and recall for this highly imbalanced classification task.



\## 🔍 Explainability with SHAP



Rather than treating the model as a black box, this project uses \*\*SHAP (SHapley Additive exPlanations)\*\* to show \*why\* each prediction was made — both globally (which features matter most overall) and locally (why one specific transaction was flagged). The top fraud-driving features (`V14`, `V12`, `V4`, `V10`, `V17`) consistently align across correlation analysis, feature importance, and SHAP values.



\## 💻 FraudLens AI — Streamlit App



An interactive web app built on top of the trained model, with three sections:



\- \*\*Dataset Overview\*\* — key stats, class distribution, and fraud patterns by hour of day

\- \*\*Transaction Analyzer\*\* — enter a transaction ID to get a live fraud prediction with confidence score and a SHAP explanation for that specific transaction

\- \*\*Model Performance\*\* — confusion matrix, ROC curve, Precision-Recall curve, and global SHAP feature importance



\## 🛠️ Tech Stack



\- \*\*Language:\*\* Python

\- \*\*Data \& ML:\*\* pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost

\- \*\*Explainability:\*\* SHAP

\- \*\*Visualization:\*\* Matplotlib, Seaborn, Plotly

\- \*\*Web App:\*\* Streamlit



\## 📂 Project Structure

FraudLens-AI/

├── notebooks/

│ ├── 01\_eda\_preprocessing.ipynb

│ ├── 02\_modeling.ipynb

│ ├── 03\_evaluation.ipynb

│ └── 04\_shap.ipynb

├── models/

│ ├── model.pkl

│ ├── amount\_scaler.pkl

│ └── time\_scaler.pkl

├── assets/

│ └── logo.png

├── app.py

├── requirements.txt

└── README.md





\## ⚙️ Running Locally



```bash

\# Clone the repo

git clone https://github.com/khushichandak27/FraudLens-AI.git

cd FraudLens-AI



\# Install dependencies

pip install -r requirements.txt



\# Download creditcard.csv from Kaggle and place it in a data/ folder

\# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud



\# Run the app

streamlit run app.py

```



\## 👤 Author



\*\*Khushi Chandak\*\*

\[GitHub](https://github.com/khushichandak27)

