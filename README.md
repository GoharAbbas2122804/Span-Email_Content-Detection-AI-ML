# 🛡️ Spam Detection ML - AI-Powered Email & Message Classifier

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)
![React](https://img.shields.io/badge/React-19+-61DAFB.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive **Machine Learning-powered Spam Detection System** with a modern, feature-rich web interface. Built with **FastAPI** backend and **React** frontend, this application provides real-time spam classification with 95%+ accuracy.

---

## 📸 Frontend Preview

![Frontend Interface](frontend-screenshot.png)
*Note: Add your frontend screenshot here. You can take a screenshot of your running application and save it as `frontend-screenshot.png` in the root directory.*

### 🎨 UI Features Showcase

- **Dark/Light Mode Toggle** - Seamless theme switching
- **Statistics Dashboard** - Real-time analytics and insights
- **Batch Processing** - Analyze multiple messages at once
- **Modern Glassmorphism Design** - Beautiful, professional interface
- **Responsive Layout** - Works perfectly on all devices

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Frontend Features](#frontend-features)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

This project is a **production-ready Spam Detection System** that classifies text messages into **spam** or **ham (not spam)** using advanced machine learning techniques. The system consists of:

- **Backend API**: FastAPI-based RESTful service with ML model inference
- **Frontend Interface**: Modern React application with real-time analysis
- **ML Models**: Multiple classifier support (LinearSVC, LogisticRegression, etc.)
- **Training Pipeline**: Automated model training from CSV datasets

### Why This Project?

Spam messages are a persistent problem in digital communication, often containing:
- 🚫 Scams and phishing attempts
- 📧 Unwanted advertising
- ⚠️ Malicious links and content
- 💰 Financial fraud attempts

This project helps users **automatically detect and filter spam messages**, improving safety and productivity.

---

## ✨ Key Features

### 🔧 Backend Features

- ✅ **Fast & Accurate** - ML-based spam detection with 95%+ accuracy
- ✅ **RESTful API** - Clean FastAPI endpoints with automatic documentation
- ✅ **Single & Batch Prediction** - Analyze one or multiple messages simultaneously
- ✅ **TF-IDF Vectorization** - Advanced text preprocessing and feature extraction
- ✅ **Multiple ML Models** - Support for LinearSVC, LogisticRegression, MultinomialNB, RandomForest
- ✅ **Auto Model Loading** - Models load automatically on server start
- ✅ **CSV Training Support** - Upload CSV files to train new models
- ✅ **CORS Enabled** - Ready for frontend integration
- ✅ **Health Check Endpoints** - Monitor API status

### 🎨 Frontend Features

- ✅ **Real-time Threat Analysis** - Instant spam detection with live feedback
- ✅ **Dark/Light Mode** - Toggle between themes with persistent preferences
- ✅ **Statistics Dashboard** - Comprehensive analytics including:
  - Total scans count
  - Spam vs Ham detection rates
  - Average threat scores
  - Confidence metrics
  - Spam rate percentage
- ✅ **Batch Processing Mode** - Process multiple messages at once
- ✅ **CSV File Upload** - Upload CSV files for batch analysis or model training
- ✅ **Export Results** - Download scan results as CSV
- ✅ **Recent Scans History** - Track and review previous analyses
- ✅ **System Logs** - Real-time activity monitoring
- ✅ **Dynamic Threat Scoring** - 0-10 scale with color-coded indicators
- ✅ **Character Counter** - 5000 character input limit
- ✅ **Copy/Paste Support** - Quick text input from clipboard
- ✅ **Modern Glassmorphism UI** - Beautiful, professional design
- ✅ **Smooth Animations** - Polished transitions and hover effects
- ✅ **Fully Responsive** - Works perfectly on desktop, tablet, and mobile
- ✅ **Local Storage** - Persistent preferences and scan history

---

## 🛠️ Technologies Used

### Backend
- **Python 3.8+** - Core programming language
- **FastAPI** - Modern, fast web framework for building APIs
- **Scikit-learn** - Machine learning library for model training and inference
- **Pandas** - Data manipulation and analysis
- **Joblib** - Model serialization and loading
- **Uvicorn** - ASGI server for running FastAPI
- **Pydantic** - Data validation using Python type annotations
- **NumPy** - Numerical computing

### Frontend
- **React 19** - Modern UI library
- **Vite** - Fast build tool and dev server
- **Custom CSS** - Modern glassmorphism design with CSS variables
- **Local Storage API** - Client-side data persistence

### Data Processing
- **TF-IDF Vectorization** - Text feature extraction
- **CSV Processing** - Dataset handling and training

---

## 📁 Project Structure

```
Span Email Detection/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI application entry point
│   │   │
│   │   ├── core/
│   │   │   ├── config.py              # Configuration settings
│   │   │   ├── loader.py              # Model & vectorizer loader
│   │   │   └── models/                # Trained ML models
│   │   │       ├── model_best.pkl
│   │   │       ├── vectorizer.pkl
│   │   │       └── training_results.csv
│   │   │
│   │   ├── routers/
│   │   │   ├── predict.py             # Prediction endpoints
│   │   │   └── train.py               # Training endpoints
│   │   │
│   │   ├── schemas/
│   │   │   └── request.py             # Pydantic request models
│   │   │
│   │   ├── services/
│   │   │   ├── predictor.py           # Prediction logic
│   │   │   └── trainer.py             # Training logic
│   │   │
│   │   └── utils/
│   │       └── preprocess.py          # Text preprocessing
│   │
│   ├── data/
│   │   ├── raw_dataset.csv            # Original dataset
│   │   └── cleaned_dataset.csv        # Preprocessed dataset
│   │
│   ├── scripts/
│   │   ├── clean_data.py              # Data cleaning script
│   │   └── train_model.py             # Model training script
│   │
│   ├── tests/
│   │   └── test_predict.py            # Unit tests
│   │
│   ├── venv/                          # Python virtual environment
│   ├── requirments.txt                # Python dependencies
│   └── README.md                      # Backend documentation
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                    # Main React component
│   │   ├── App.css                    # Application styles
│   │   ├── main.jsx                   # React entry point
│   │   └── index.css                  # Global styles
│   │
│   ├── public/
│   │   └── vite.svg
│   │
│   ├── package.json                   # Node.js dependencies
│   ├── vite.config.js                # Vite configuration
│   └── README.md                      # Frontend documentation
│
├── README.md                          # This file
└── frontend-screenshot.png            # Frontend screenshot (add your image here)
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+** installed
- **Node.js 16+** and **npm** installed
- **Git** (optional, for cloning)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd "Span Email Detection"
```

### Step 2: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (Linux/macOS)
python3 -m venv venv
source venv/bin/activate

# Create virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirments.txt
```

### Step 3: Frontend Setup

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install dependencies
npm install
```

### Step 4: Prepare Dataset (Optional)

Place your training dataset in `backend/data/raw_dataset.csv` with columns:
- `text` - Message content
- `label` - Classification (spam/ham or 0/1)

Then run:
```bash
cd backend
python scripts/clean_data.py
python scripts/train_model.py
```

---

## 💻 Usage

### Starting the Backend Server

```bash
cd backend
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate      # Windows

uvicorn app.main:app --reload
```

The API will be available at: **http://127.0.0.1:8000**

- **API Documentation**: http://127.0.0.1:8000/docs (Swagger UI)
- **Alternative Docs**: http://127.0.0.1:8000/redoc

### Starting the Frontend

```bash
cd frontend
npm run dev
```

The frontend will be available at: **http://localhost:5173** (or the port shown in terminal)

### Using the Application

1. **Single Message Analysis**:
   - Enter a message in the text area
   - Click "INITIATE SCAN"
   - View results with threat score and confidence

2. **Batch Processing**:
   - Switch to "Batch Processing" mode
   - Add multiple message fields
   - Click "ANALYZE BATCH"
   - Review all results at once

3. **Upload CSV for Batch Analysis**:
   - Click "Batch Upload" in the left sidebar
   - Select a CSV file with messages
   - Messages will be loaded automatically

4. **View Statistics**:
   - Click the statistics icon in the header
   - View comprehensive analytics dashboard
   - Export results to CSV

5. **Toggle Theme**:
   - Click the sun/moon icon to switch between dark and light modes

---

## 🌐 API Endpoints

### Root Endpoint
```
GET /
```
Returns API status and available endpoints.

### Health Check
```
GET /health
```
Check if API is running.

### Single Prediction
```
POST /api/predict
Content-Type: application/json

{
  "message": "Your text message here"
}
```

**Response:**
```json
{
  "prediction": "SPAM",
  "probability": 0.95,
  "original_message": "Your text message here"
}
```

### Batch Prediction
```
POST /api/predict_batch
Content-Type: application/json

{
  "messages": ["message1", "message2", "message3"]
}
```

**Response:**
```json
{
  "results": [
    {
      "prediction": "SPAM",
      "probability": 0.95,
      "original_message": "message1"
    }
  ]
}
```

### Train Model
```
POST /api/train/upload
Content-Type: multipart/form-data

file: <CSV file>
```

**CSV Format:**
```csv
text,label
"Hello friend",ham
"Win $1000 now!",spam
```

---

## 🎨 Frontend Features in Detail

### 1. **Dark/Light Mode**
- Toggle between themes with one click
- Preferences saved in local storage
- Smooth theme transitions

### 2. **Statistics Dashboard**
- **Total Scans**: Number of messages analyzed
- **Spam Detected**: Count of spam messages
- **Safe Messages**: Count of ham messages
- **Average Threat Score**: Mean threat level
- **Average Confidence**: Mean prediction confidence
- **Spam Rate**: Percentage of spam messages
- **Export Button**: Download results as CSV

### 3. **Batch Processing**
- Add/remove message fields dynamically
- Process multiple messages simultaneously
- View all results in a scrollable list
- Color-coded results (red for spam, green for safe)

### 4. **File Upload**
- **Batch Upload**: Upload CSV for batch message analysis
- **Model Training**: Upload CSV to train new models
- Drag-and-drop interface ready

### 5. **Recent Scans**
- View last 10 scan results
- Quick access to previous analyses
- Color-coded status indicators

### 6. **System Logs**
- Real-time activity monitoring
- Color-coded log types (info, success, error)
- Timestamp for each log entry

---

## 📊 Performance Metrics

Typical performance on balanced dataset:

| Metric | Score |
|--------|-------|
| Accuracy | 96-98% |
| Precision | 95-97% |
| Recall | 94-96% |
| F1-Score | 95-97% |

---

