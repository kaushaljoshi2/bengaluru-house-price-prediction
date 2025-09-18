# Bengaluru House Price Prediction
This project implements a machine learning model to predict house prices in Bengaluru, India, based on various property features. The dataset is cleaned, engineered, and used to train a Linear Regression model that estimates prices given location, size, number of bedrooms (BHK), and bathrooms.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Cleaning & Feature Engineering](#data-cleaning--feature-engineering)
4. [Model Training](#model-training)
5. [Usage](#usage)
6. [Future Work](#future-work)
7. [Acknowledgments](#acknowledgments)

## Project Overview

Real estate pricing is complex and influenced by many factors. This project aims to build a reliable house price predictor using Bengaluru’s housing data by:

- Cleaning and preprocessing raw data
- Removing outliers and irrelevant columns
- Creating new features like price per square foot
- Encoding categorical variables like location
- Training and evaluating a Linear Regression model
- Providing a function to predict prices based on user input

## Dataset

The dataset used is Bengaluru_House_Data.csv which contains property listings with features like:

- Location
- Size (number of bedrooms)
- Total square footage
- Number of bathrooms
- Price (in lakhs)

📌 **Original dataset link:**
[Bengaluru House Data](https://www.kaggle.com/datasets/mohitpratap166/bengaluru-house-data)

Please refer to the Kaggle page for licensing, usage restrictions, and further dataset documentation.

## Data Cleaning & Feature Engineering

Key steps performed:

- Removed irrelevant columns: area_type, availability, balcony, society
- Handled missing values by discarding incomplete rows
- Extracted the number of bedrooms (BHK) from the size column
- Converted ranges in total_sqft (like "2100-2850") into their average numeric value
- Created a new feature price_per_sqft for identifying price-based outliers
- Grouped rare location entries (≤ 10 occurrences) into a common category other
- Removed properties with unrealistic square feet per bedroom (very low values)
- Removed outliers in price per square foot on a per-location basis
- Filtered out entries where number of bathrooms was significantly more than bedrooms (bathrooms > BHK + 2)
- One-hot encoded the location categorical feature

## Model Training

**Features**: `total_sqft`, `bath`, `BHK`, and one-hot encoded location columns
**Target**: `price`
**Model**: Linear Regression
**Validation**: 80-20 train-test split + ShuffleSplit cross-validation (5 folds)
**Evaluation Metric**: R² score (coefficient of determination)

## Usage

You can use the model’s prediction function as follows:

`price_predict(location, sqft, bath, BHK)`

Example:

`price_predict('1st Phase JP Nagar', 1000, 2, 2)`

This returns the predicted house price for a 1000 sqft home, 2 bathrooms, 2 BHK in 1st Phase JP Nagar (in lakhs, as per the data’s units).

## Future Work
- Experiment with more complex models (e.g., Random Forest, XGBoost, or neural networks)
- Add more features: property age, amenities, proximity to public transport, etc.
- Better handling of location data — clustering nearby localities perhaps
- Build an interactive user interface (web app) for price prediction

## Acknowledgments
📊 Dataset: [Bengaluru House Data by Mohit Pratap on Kaggle](https://www.kaggle.com/datasets/mohitpratap166/bengaluru-house-data) 
🙏 Thank you to Kaggle for providing the data and platform.
