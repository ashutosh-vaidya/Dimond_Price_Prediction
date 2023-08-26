# Dimond_Price_Prediction
End-to-end ML project on Dimond Price Prediction

## Problem Statement :

You are hired by a company Gem Stones co ltd, which is a cubic zirconia manufacturer. You are provided with the dataset containing the prices and other attributes of almost 27,000 cubic zirconia (which is an inexpensive diamond alternative with many of the same qualities as a diamond). The company is earning different profits on different prize slots. You have to help the company in predicting the price for the stone on the basis of the details given in the dataset so it can distinguish between higher profitable stones and lower profitable stones so as to have a better profit share. Also, provide them with the best 5 attributes that are most important.

****
## Approach :

This project was originally created as a part of follow along project taught by **Krish Naik** as part of iNeuron course FSDS 2.0 and then enhance upon at my own pace by doing detailed EDA, exploring the different ways to create user interface (original approach was a flask app).

### Steps followed during the project:

**1. Data Collection :**

The data set was provide as a part of Kaggle competition and can be downloaded from [here](https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv)

**2. Exploratory Data Analysis**
**3. Data Preprocessing**
**4. Feature Engineering**
**5. Model Building**
**6. Model Evaluation**

A detailed EDA is done in Jupyter Notebook and can be open using below,

<a target="_blank" href="https://colab.research.google.com/github/ashutosh-vaidya/Dimond_Price_Prediction/blob/main/notebooks/EDA.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Based on the EDA, data cleaning, pre-processing and feature engineering performed. Model is then trained with **Linear Regression, Lasso, Ridge, Elastic Net** algorithm and then evaluated using **R2 Score** and the best performing model is selected.

Below table show the model performance:

| **Algorithm** | **RMSE** | **MAE** | **R2 Score** |
|-----|-----|-----|-----|
| Linear Regression | 1013.904 | 674.025 | 93.68 |
| Lasso | 1013.878 | 675.071 | 93.68 |
| Ridge | 1013.905 | 674.055 | 93.68 |
| Elastic Net | 1533.416 | 1060.736 | 85.564 |

**7. Modular Code** is then written to recreate all the work performed in notebook using creating different pipelines for data transformation, data pre-processing, model training and generating pickle file to be used for predict the values with proper logging and exception handling  

**8. User Interface** 

Created using Gradio and solution is deployed on spaces provided by hugging face.

****
## Tools and Tech Used :

- Python Libraries : pandas, NumPy, matplotlib, seaborn, scikit-learn
- Visual Studio Code
- [Hugging Face - spaces](https://huggingface.co/spaces) - Used to host the application
- [Gradio](https://www.gradio.app/) - Used for creating user interface

****
## Live Demo :

[Diamond Price Prediction](https://huggingface.co/spaces/imashutosh/diamond_price_prediction)

Some Sample values which can be used to input into the inputs

| **Carat** | **Depth** | **Table** | **X** | **Y** | **Z** | **Cut** | **Color** | **Clarity** | **Actual Price** |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|0.71|61.4|56|5.74|5.77|3.53|Ideal|D|VS2|3510|
|2|59.5|57|8.08|8.15|4.89|Very Good|G|SI2|14691|
|1.52|60.8|59|7.36|7.4|4.49|Premium|G|SI2|9970|
|1.5|60.1|61|7.45|7.42|4.47|Good|H|IF|12641|

****

## References : 

- To get some domain knowledge and understanding: [https://www.diamonds.pro/education/4cs-diamonds/](https://www.diamonds.pro/education/4cs-diamonds/)
- Kaggle submission by [SERGEY SAHAROVSKIY](https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e8-2023-eda-and-submission)
- Gradio [documentation](https://www.gradio.app/docs/interface)
- Hugging Face [deployment how to](https://huggingface.co/docs/hub/spaces-overview)
