# FastAPI Customer Churn Prediction

A complete machine learning application built with FastAPI that predicts customer subscription churn. Features a modern web interface for both single and batch predictions.

## 🚀 Features

- **Point Prediction**: Predict churn for individual customers with interactive form
- **Batch Prediction**: Upload CSV files for bulk predictions with downloadable results
- **Modern UI**: Clean, responsive interface with teal/cyan brand colors
- **RESTful API**: Well-structured endpoints for easy integration
- **Simple ML Pipeline**: Basic data cleaning and feature engineering

## 📁 Project Structure

```
fastAPI_customer_churn/
├── data/
│   └── customer_subscription_churn_dataset.csv
├── frontend/
│   └── index.html
├── models/
│   └── churn_model.pkl
├── downloads/
├── venv/
├── create_model.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fastapi-customer-churn-prediction.git
cd fastapi-customer-churn-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
```

3. **Activate virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Mac/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
```bash
pip install fastapi uvicorn pandas numpy scikit-learn python-multipart jinja2
```

5. **Generate requirements.txt** (optional)
```bash
pip freeze > requirements.txt
```

## 🎯 Usage

### Train the Model

Run the model training script first:
```bash
python create_model.py
```
This will:
- Load and clean the data
- Perform feature engineering
- Train the model
- Save it to `models/churn_model.pkl`

### Start the Application

```bash
 uvicorn main:app --reload
```

The application will start at: `http://localhost:8000`

### API Endpoints

#### 1. Home Page
```
GET /
```
Serves the frontend interface

#### 2. Single Prediction
```
POST /predict
Content-Type: application/json

{
  "tenure_months": 29,
  "plan_type": "Premium",
  "region": "Central",
  "avg_sessions_per_week": 6.11,
  "avg_session_duration_minutes": 52.9,
  "feature_usage_score": 29.9,
  "days_since_last_login": 29,
  "monthly_fee": 786,
  "payment_method": "NetBanking",
  "failed_payments_last_3_months": 1,
  "discount_applied": "No",
  "support_tickets_count": 0,
  "avg_ticket_resolution_time_hours": 7.3,
  "customer_satisfaction_score": 5,
  "marketing_email_opens_last_30_days": 4
}
```

#### 3. Batch Prediction
```
POST /batch-predict
Content-Type: multipart/form-data

file: <csv_file>
```
Returns a CSV file with predictions

## 📊 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| tenure_months | Numeric | Customer tenure in months |
| plan_type | Categorical | Subscription plan (Premium, PRO, Pro) |
| region | Categorical | Customer region (Central, North, South, West, East) |
| avg_sessions_per_week | Numeric | Average sessions per week |
| avg_session_duration_minutes | Numeric | Average session duration |
| feature_usage_score | Numeric | Feature usage score (0-100) |
| days_since_last_login | Numeric | Days since last login |
| monthly_fee | Numeric | Monthly subscription fee |
| payment_method | Categorical | Payment method (NetBanking, UPI, CreditCard, DebitCard) |
| failed_payments_last_3_months | Numeric | Failed payment count |
| discount_applied | Categorical | Discount status (Yes/No) |
| support_tickets_count | Numeric | Support tickets raised |
| avg_ticket_resolution_time_hours | Numeric | Average ticket resolution time |
| customer_satisfaction_score | Numeric | Satisfaction score (1-5) |
| marketing_email_opens_last_30_days | Numeric | Email opens in last 30 days |

## 💻 Frontend Features

### Point Prediction
- Interactive form with dropdowns for categorical features
- Number inputs with min/max hints for numerical features
- Real-time prediction results
- Clean, modern design

### Batch Prediction
- File upload interface
- Automatic CSV processing
- Downloadable results file
- Progress indication

## 🎨 Brand Colors

- Primary: Teal (#00CED1)
- Secondary: Dark Cyan (#008B8B)
- Background: Dark (#1a1a2e)
- Accent: Light Cyan for highlights

## 🔧 Technology Stack

- **Backend**: FastAPI
- **ML Framework**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Server**: Uvicorn
- **Frontend**: HTML5, CSS3, Vanilla JavaScript

## 📝 Model Details

- **Algorithm**: Classification (LogisticRegression/RandomForest)
- **Features**: 15 input features
- **Target**: Binary churn prediction (0 = No Churn, 1 = Churn)
- **Preprocessing**: Basic cleaning and feature engineering

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Rakesh**
- AI Instructor & Course Developer
- Specializing in AI Engineering, ML, and Data Analytics

## 🙏 Acknowledgments

- Built with FastAPI framework
- Powered by Scikit-learn
- Developed for educational purposes

---

**Note**: This application is designed for demonstration and educational purposes. For production use, consider additional features like authentication, logging, monitoring, and enhanced error handling.
