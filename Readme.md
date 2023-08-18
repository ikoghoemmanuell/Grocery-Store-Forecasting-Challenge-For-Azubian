# Grocery Sales Forecasting Challenge for Azubian

[![View Repositories](https://img.shields.io/badge/View-My_Repositories-blue?logo=GitHub)](https://github.com/ikoghoemmanuell?tab=repositories)
[![View My Profile](https://img.shields.io/badge/MEDIUM-Article-purple?logo=Medium)](https://github.com/ikoghoemmanuell/Grocery-Store-Forecasting-Challenge-For-Azubian/blob/main/Article.md)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-yellow)](https://huggingface.co/spaces/ikoghoemmanuell/SEER-A_sales_forecasting_app)
[![Website](https://img.shields.io/badge/My-Website-darkgreen)](https://emmanuelikogho.netlify.app/)

Increase sales of groceries using exploratory data analysis and machine learning.

![grocery-sales-challenge-readme-azubian-](https://github.com/ikoghoemmanuell/Grocery-Store-Forecasting-Challenge-For-Azubian/assets/102419217/88f2040f-72a1-4bb0-936a-f2520ae73a2f)

## Introduction

This repository serves as a case study for the Grocery Store Forecasting Challenge for Azubian on Zindi Africa. The challenge focuses on predicting the sales of various products in grocery stores based on historical data. This case study explores the data, methodologies, and models used to tackle the challenge, providing insights into the predictive analytics process.

## Dataset

The dataset used for this case study is provided on the Zindi Africa platform. It consists of historical sales data, product information, and store information. The dataset is utilized to build models that can accurately forecast future sales and help grocery stores optimize their inventory management and supply chain operations.

## Setup

Fork this repository and run the notebook on Colab.

Learn about Google Colab here.
Learn how to connect Colab to your github account here.

## Methodology

1. Exploratory Data Analysis (EDA): The case study begins with an in-depth exploration of the dataset to understand its structure, variables, and patterns. EDA techniques such as data visualization and statistical analysis are applied to gain insights into the sales patterns and relationships between variables.

2. Feature Engineering: The dataset is preprocessed and transformed to create meaningful features that capture relevant information for sales forecasting. This involves tasks such as handling missing values, encoding categorical variables, and creating lagged features to account for time dependencies.

3. Model Development: Various machine learning and time series forecasting models are developed and evaluated to identify the best-performing approach. This may include traditional regression models, ensemble methods, or advanced techniques specifically designed for time series forecasting, such as ARIMA, SARIMA, or Prophet.

4. Model Evaluation: The performance of the developed models is assessed using appropriate evaluation metrics such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE). The models are compared based on their predictive accuracy and ability to capture the underlying sales patterns.

5. Forecasting and Visualization: The selected model is used to generate sales forecasts for future time periods. The forecasted results are visualized using graphs and charts to provide actionable insights and aid decision-making for grocery store management.

## Repository Structure

The repository is organized as follows:

- `data/`: Contains the dataset files.
- `notebooks/`: Contains Jupyter notebooks showcasing the step-by-step implementation of the case study, including EDA, feature engineering, model development, and evaluation.
- `dev/`: Contains any source code or scripts used in the case study, such as data preprocessing or custom functions.

Feel free to explore the notebooks and source code to gain a deeper understanding of the case study methodology.

# to run the app

Fork this repository first of all. Now follow the steps below

You need to have [`Python 3`](https://www.python.org/) on your system (**a Python version lower than 3.10**). Then you can clone this repo and being at the repo's `root :: repository_name> ...` follow the steps below:

- Windows:

```python
python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt
```

- Linux & MacOs:

```python
python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt
```

The both long command-lines have a same structure, they pipe multiple commands using the symbol `;` but you may manually execute them one after another.

1. **Create the Python's virtual environment** that isolates the required libraries of the project to avoid conflicts;
2. **Activate the Python's virtual environment** so that the Python kernel & libraries will be those of the isolated environment;
3. **Upgrade Pip, the installed libraries/packages manager** to have the up-to-date version that will work correctly;

change directory to where the app is located

```python
cd dev; cd app
```

to run app, install the requirements for the app,

```python
pip install streamlit transformers torch
```

then go to your terminal and run

```python
streamlit run app.py
```

# screenshot

![seer-sales-prediction-for-azubian](https://github.com/ikoghoemmanuell/Grocery-Store-Forecasting-Challenge-For-Azubian/assets/102419217/933f5f5b-2976-4499-bd48-050c1bab5bd0)


## 👏 Support

If you found this article helpful, please give it a clap or a star on GitHub!

## Author

- [Emmanuel Ikogho](https://www.linkedin.com/in/emmanuel-ikogho/)
