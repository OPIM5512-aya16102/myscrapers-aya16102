📊 Used Car Price Prediction & Backtesting Pipeline

This project builds and evaluates machine learning models to predict used car prices using structured listing data. It includes a full end-to-end pipeline for data cleaning, feature engineering, time-based backtesting, and model evaluation.

The system trains multiple regression models (Decision Tree, Random Forest, and XGBoost), applies log-transformed targets for stability, and evaluates performance using rolling time-based splits to simulate real-world prediction scenarios.

In addition to standard metrics (MAE, RMSE, R², MAPE, bias), the pipeline generates model interpretability outputs including permutation feature importance and partial dependence plots (PDPs). All artifacts (models, predictions, and diagnostics) are saved locally and uploaded to Google Cloud Storage for tracking and monitoring over time.

This setup is designed for production-style experimentation, enabling continuous evaluation of model performance as new data becomes available.
