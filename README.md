# 🚗 Craigslist Car Price Prediction using GenAI + Machine Learning

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ahmed-ahmed-831765190/)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![GCP](https://img.shields.io/badge/Google_Cloud-GCP-1a73e8.svg)
![Vertex AI](https://img.shields.io/badge/Vertex_AI-GenAI-fcb900.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)

## 🎯 Purpose
The goal of this project is to accurately predict the market price of used vehicles listed on Craigslist. By combining **Generative AI** for unstructured data extraction with traditional **Machine Learning** models, this pipeline tracks automotive market trends and provides reliable price estimations.

## ⚙️ Workflows & Tech Stack
The entire pipeline—from data ingestion to model deployment—is fully automated and hosted on **Google Cloud Platform (GCP)**.

*   **Data Scraping & Extraction:** Powered by **Vertex AI** (LLMs), which intelligently parses unstructured Craigslist descriptions and extracts clean, tabular data.
*   **Model Training:** Evaluates multiple regressors (Decision Trees, Random Forests, XGBoost) using K-Fold cross-validation and hyperparameter tuning.
*   **Deployment:** Artifacts, predictions, and plots are saved directly to **Google Cloud Storage (GCS)**, enabling a seamless ML pipeline.

## 📊 View Live Results & Dashboards
You can interactively explore how the models perform, compare error metrics, and see exactly which features (like Age, Mileage, or Make) drive the price of a car using the interactive UI.

**To view the interactive dashboards:**
1. Download the notebook file: [`Model_Trending_Notebook.ipynb`](./Model_Trending_Notebook.ipynb) *(Make sure this file is in your repo!)*
2. Open the file in **Google Colab** (recommended) or your local Jupyter environment.
3. Run the cells to launch the interactive widgets, which include:
   * **Model Performance Scrubber:** Compare MAE, RMSE, and R² over time.
   * **Feature Importance Dashboard:** See what vehicle features matter most to the algorithm.
   * **Partial Dependence Plots (PDP):** Visualize how specific variables (like car age) directly impact the predicted dollar amount.

---

## 📬 Let's Connect
Created by **Ahmed Ahmed**. If you have any questions about this project, the GenAI integration, or the GCP deployment pipeline, feel free to reach out!
