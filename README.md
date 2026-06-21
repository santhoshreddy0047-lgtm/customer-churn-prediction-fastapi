# Customer Churn Prediction System using FastAPI & Machine Learning

A complete end-to-end Machine Learning application that predicts customer churn using customer subscription and engagement data. The application provides both real-time predictions through a web interface and batch predictions using CSV uploads.

## Features

* Real-time customer churn prediction
* Batch prediction using CSV file uploads
* Interactive web-based dashboard
* Machine Learning model training and evaluation
* RESTful API built with FastAPI
* Downloadable prediction results
* Responsive frontend interface

## Project Structure

```text
customer-churn-prediction-fastapi/
├── data/
├── frontend/
│   └── index.html
├── models/
│   └── churn_model.pkl
├── create_model.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Technologies Used

* Python
* FastAPI
* Scikit-learn
* Pandas
* NumPy
* HTML5
* CSS3
* JavaScript
* Git & GitHub

## Machine Learning Workflow

1. Data Cleaning and Preprocessing
2. Feature Engineering
3. Model Training using Random Forest Classifier
4. Model Evaluation
5. Model Serialization using Pickle
6. API Deployment using FastAPI

## Model Performance

* Accuracy: ~75%
* Classification Type: Binary Classification
* Target Variable: Customer Churn (Yes/No)

## Installation

```bash
git clone https://github.com/santhoshreddy0047-lgtm/customer-churn-prediction-fastapi.git

cd customer-churn-prediction-fastapi

pip install -r requirements.txt
```

## Train the Model

```bash
python create_model.py
```

## Run the Application

```bash
python main.py
```

or

```bash
uvicorn main:app --reload
```

Application URL:

```text
http://127.0.0.1:8000
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

## Key Features Used for Prediction

* Tenure Months
* Plan Type
* Region
* Monthly Fee
* Customer Satisfaction Score
* Average Session Duration
* Days Since Last Login
* Feature Usage Score
* Payment Method
* Marketing Email Opens

## Future Improvements

* Improve model performance using XGBoost
* Add authentication and user management
* Deploy application on Render or Railway
* Add dashboards and visual analytics
* Implement model monitoring

## Author

**Santhosh Reddy**

Data Scientist skilled in Python, SQL, Machine Learning, Power BI, FastAPI, and Data Analytics.

GitHub:
https://github.com/santhoshreddy0047-lgtm

```
```
