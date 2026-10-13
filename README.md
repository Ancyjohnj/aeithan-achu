##Car Price Prediction Model

Name: Ancy John J.

Organization: Entri Elevate

Date: 12 October 2026

#31. Overview of Problem Statement

Car prices depend on several factors, such as engine size, horsepower, curb weight, fuel

type, and car dimensions. Predicting car prices accurately helps automobile companies

understand the factors affecting vehicle pricing.

This project uses machine learning regression algorithms to predict car prices based on

vehicle features.

##2. Objective

To develop the best-performing machine learning regression model for predicting car prices

and identify the important features affecting car prices.

##3. Data Description

Dataset: CarPrice_Assignment.csv

Source: Dataset provided for the project

Target Variable: Price

Dataset Size: 205 rows and 26 columns before preprocessing

Features: Engine size, horsepower, curb weight, highway MPG, city MPG, car width, car

length, wheelbase, fuel type, aspiration, car body, drive wheel, engine type, and other

vehicle-related features.

##4. Data Collection

The dataset was loaded into a Jupyter Notebook using Pandas. The data was examined using

head(), shape, info(), and descriptive statistics to understand its structure and

characteristics.

##5. Data Preprocessing – Data Cleaning

Checked the dataset structure and data types.

Removed the unnecessary car_ID column.

Processed the CarName column.

Checked for missing values and duplicate records.

Examined numerical features for outliers and skewness.

Prepared the dataset for machine learning.

##6. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the relationships between vehicle

features and car prices.

The visualizations included:

Feature importance graph

Model comparison graph

Actual vs Predicted Price graph

These visualizations helped understand model performance and the important variables

affecting car prices.

##7. Feature Engineering

Categorical variables were converted into numerical values using one-hot encoding with

Pandas get_dummies(). The target variable price was separated from the input features.

##8. Feature Selection

Random Forest feature importance was used to identify significant variables affecting car

prices.

The most important features included:

Engine Size

Curb Weight

Highway MPG

Horsepower

Car Width

Engine size and curb weight were the most influential features in the trained Random Forest

model.

##9. Split Data into Training and Testing Sets

The dataset was divided into training and testing sets using an 80:20 ratio.

Training data: 80%

Testing data: 20%

Random state: 42

##10. Feature Scaling

StandardScaler was used for the Support Vector Regressor (SVR) to standardize the input

features. The scaler was fitted on the training data and then applied to the testing data.

##11. Build the ML Model

Five regression algorithms were implemented:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

Support Vector Regressor (SVR)

All five models were trained using the training dataset.

##12. Model Evaluation

The models were evaluated using R-squared (R²), Mean Squared Error (MSE), and Mean Absolute

Error (MAE).

Model

R² Score

MAE

MSE

Linear Regression

0.916

1733.39

6600950.84

Decision Tree Regressor

0.908

1798.48

7276978.44

Random Forest Regressor

0.958

1287.90

3333544.23

Gradient Boosting Regressor

0.928

1692.53

5708107.41

Support Vector Regressor

-0.101

5701.44

86890173.85

Best Model: Random Forest Regressor

The Random Forest model achieved the highest R² score and the lowest MAE and MSE among the five models. Therefore, it was selected as the best-performing model for this dataset.

Hyperparameter Tuning and Pipeline
GridSearchCV with 5-fold cross-validation was used to tune the Random Forest Regressor.

Best Parameters:

n_estimators: 100

max_depth: None

min_samples_split: 2

The R² score remained approximately 0.958 after tuning. The model performance did not improve because the selected parameters were the same as those of the original model.

Save the Model
The trained Random Forest model and its feature column names were saved using Joblib.

car_price_model.pkl

feature_columns.pkl

These files can be used to load the model and prepare input features for future predictions.

Test with Unseen Data
The trained model was evaluated on the test dataset, which was not used during model training. The model was also prepared for saving and future integration into a prediction application.

Interpretation of Results (Conclusion)
The project compared five regression algorithms for predicting car prices. Random Forest Regressor performed best, with an R² score of approximately 0.958 and an MAE of 1287.90.

Feature importance analysis showed that engine size and curb weight were the most significant features affecting car prices.

Hyperparameter tuning was performed using GridSearchCV, but the model's performance remained unchanged. The results indicate that Random Forest Regressor is a suitable model for this dataset.

One limitation is that the dataset contains only 205 records, so the results may not represent every car or market.

Future Work
Collect more car pricing data.

Explore additional feature selection techniques such as SelectKBest.

Experiment with other regression algorithms and hyperparameters.

Build and deploy a user-friendly web application using Flask.

Evaluate the model on additional unseen car data.

Update the model when new data becomes available.
