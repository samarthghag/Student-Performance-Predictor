# 🎓 Student Performance Predictor

A comprehensive Streamlit web application that predicts student academic performance using machine learning. This application analyzes various student attributes and predicts whether a student will pass or fail based on historical data.

## 📋 Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### 📊 Data Overview
- **Dataset Statistics**: View total students, features, pass rate, and average grades
- **Data Sample**: Interactive display of the dataset
- **Missing Value Analysis**: Comprehensive check for data quality
- **Basic Statistics**: Descriptive statistics for all numerical features

### 📈 Data Analysis & Visualizations
- **Grade Distribution**: Histogram showing the distribution of final grades
- **Pass/Fail Analysis**: Pie chart visualization of student outcomes
- **Gender Performance**: Comparative analysis of performance by gender
- **Study Time Impact**: Scatter plot showing correlation between study time and grades
- **Family Support Analysis**: Impact of family support on academic performance
- **Internet Access Impact**: How internet access affects student grades
- **Feature Correlation Heatmap**: Interactive correlation matrix of all numerical features

### 🤖 Machine Learning Model
- **Random Forest Classifier**: Robust ensemble learning algorithm
- **Model Evaluation**: Comprehensive performance metrics including:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
  - Feature Importance Analysis
- **Cross-validation**: Train/test split for reliable performance estimation

### 🔮 Prediction Interface
- **Interactive Input Form**: User-friendly interface to input student information
- **Real-time Predictions**: Instant pass/fail predictions with probability scores
- **Probability Visualization**: Visual representation of prediction confidence
- **Comprehensive Features**: Input fields for all relevant student attributes

## 📊 Dataset

The application uses the `student_mat.csv` dataset containing information about student performance in mathematics. The dataset includes:

### Student Demographics
- **school**: Student's school (GP or MS)
- **sex**: Student's gender (F or M)
- **age**: Student's age (15-22)
- **address**: Home address type (Urban or Rural)
- **famsize**: Family size (≤3 or >3)

### Family Background
- **Pstatus**: Parent's cohabitation status (Together or Apart)
- **Medu**: Mother's education level (0-4)
- **Fedu**: Father's education level (0-4)
- **Mjob**: Mother's job
- **Fjob**: Father's job
- **guardian**: Student's guardian

### Academic Information
- **reason**: Reason to choose this school
- **traveltime**: Home to school travel time
- **studytime**: Weekly study time
- **failures**: Number of past class failures
- **schoolsup**: Extra educational support
- **famsup**: Family educational support
- **paid**: Extra paid classes
- **higher**: Wants to take higher education

### Social & Lifestyle
- **activities**: Extra-curricular activities
- **nursery**: Attended nursery school
- **internet**: Internet access at home
- **romantic**: In a romantic relationship
- **famrel**: Quality of family relationships (1-5)
- **freetime**: Free time after school (1-5)
- **goout**: Going out with friends (1-5)
- **Dalc**: Workday alcohol consumption (1-5)
- **Walc**: Weekend alcohol consumption (1-5)
- **health**: Current health status (1-5)
- **absences**: Number of school absences

### Target Variables
- **G1**: First period grade (0-20)
- **G2**: Second period grade (0-20)
- **G3**: Final grade (0-20) - **Primary target for prediction**

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step-by-Step Installation

1. **Clone or Download the Repository**
   ```bash
   git clone <repository-url>
   cd studentmarkspredicition
   ```

2. **Create a Virtual Environment** (Recommended)
   ```bash
   python -m venv .venv
   ```

3. **Activate the Virtual Environment**
   ```bash
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

4. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

### Running the Application

1. **Navigate to the Project Directory**
   ```bash
   cd studentmarkspredicition
   ```

2. **Run the Streamlit Application**
   ```bash
   streamlit run app.py
   ```

3. **Access the Application**
   - The application will automatically open in your default web browser
   - If not, navigate to `http://localhost:8501`

### Using the Application

#### 1. Data Overview Page
- View dataset statistics and basic information
- Explore data quality and missing values
- Review descriptive statistics

#### 2. Data Analysis Page
- Explore interactive visualizations
- Analyze relationships between different features
- Understand patterns in student performance

#### 3. Model Training Page
- Train the Random Forest classifier
- View model performance metrics
- Analyze feature importance
- Review confusion matrix and classification report

#### 4. Prediction Page
- Input student information using the interactive form
- Get real-time predictions with probability scores
- View prediction confidence through visualizations

## 📁 Project Structure

```
studentmarkspredicition/
│
├── app.py                 # Main Streamlit application
├── student_mat.csv        # Student dataset
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .venv/                # Virtual environment (created after setup)
└── __pycache__/          # Python cache files (auto-generated)
```

## 🤖 Model Details

### Algorithm: Random Forest Classifier

**Why Random Forest?**
- **Robust Performance**: Handles both numerical and categorical features effectively
- **Feature Importance**: Provides insights into which factors most influence student performance
- **Overfitting Resistance**: Ensemble method reduces overfitting risk
- **No Feature Scaling Required**: Works well with features of different scales

### Model Configuration
- **n_estimators**: 100 decision trees
- **random_state**: 42 (for reproducible results)
- **train_test_split**: 80% training, 20% testing

### Performance Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: Ratio of true positive predictions
- **Recall**: Ratio of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions vs actual results

### Feature Engineering
- **Target Variable**: Pass/Fail based on final grade (G3 ≥ 10 = Pass)
- **Categorical Encoding**: Label encoding for categorical variables
- **Feature Selection**: Excludes G1, G2, G3 grades to avoid data leakage

## 📱 Screenshots

*Note: Add screenshots of your application here when running*

### Data Overview Page
- Dashboard with key metrics
- Dataset preview table
- Missing value analysis

### Data Analysis Page
- Interactive charts and graphs
- Correlation heatmap
- Performance analysis by different factors

### Model Training Page
- Model performance metrics
- Feature importance visualization
- Confusion matrix

### Prediction Page
- Input form interface
- Prediction results with probabilities
- Visual probability display

## 🛠️ Customization

### Adding New Features
1. Update the `student_mat.csv` dataset with new columns
2. Modify the `preprocess_data()` function in `app.py`
3. Update the prediction interface in `show_prediction_interface()`

### Changing the Model
1. Import your preferred algorithm from scikit-learn
2. Modify the `train_model()` function
3. Update model parameters as needed

### Styling Customizations
- Modify the CSS in the `st.markdown()` section
- Change color schemes in Plotly visualizations
- Update page layout and components

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions, suggestions, or support:
- Email: [your-email@example.com]
- GitHub: [your-github-username]

## 🙏 Acknowledgments

- **Dataset**: Student Performance Dataset (Mathematics)
- **Streamlit**: For the amazing web app framework
- **Plotly**: For interactive visualizations
- **scikit-learn**: For machine learning capabilities
- **Pandas & NumPy**: For data manipulation and analysis

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ using Streamlit and Python**

*Empowering educators with data-driven insights for student success!*
