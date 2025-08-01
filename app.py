import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the student dataset"""
    try:
        df = pd.read_csv('student_mat.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset file 'student_mat.csv' not found. Please ensure the file is in the same directory.")
        return None

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Create target variable (Pass/Fail based on final grade G3)
    # Pass if G3 >= 10 (typical passing grade)
    data['Pass'] = (data['G3'] >= 10).astype(int)
    
    # Encode categorical variables
    categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                       'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 
                       'nursery', 'higher', 'internet', 'romantic']
    
    label_encoders = {}
    for col in categorical_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
    
    return data, label_encoders

def train_model(data):
    """Train a Random Forest classifier"""
    # Features (excluding G1, G2, G3 as they are the grades we want to predict)
    feature_cols = [col for col in data.columns if col not in ['G1', 'G2', 'G3', 'Pass']]
    X = data[feature_cols]
    y = data['Pass']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X_train, X_test, y_train, y_test, y_pred, accuracy, feature_cols

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 Student Performance Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Data Overview", "Data Analysis", "Model Training", "Prediction"])
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Preprocess data
    data, label_encoders = preprocess_data(df)
    
    if page == "Data Overview":
        show_data_overview(df, data)
    elif page == "Data Analysis":
        show_data_analysis(df, data)
    elif page == "Model Training":
        show_model_training(data)
    elif page == "Prediction":
        show_prediction_interface(data, label_encoders)

def show_data_overview(df, data):
    """Display data overview and basic statistics"""
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        pass_rate = (data['Pass'].sum() / len(data) * 100)
        st.metric("Pass Rate", f"{pass_rate:.1f}%")
    with col4:
        avg_grade = df['G3'].mean()
        st.metric("Average Final Grade", f"{avg_grade:.2f}")
    
    # Display raw data
    st.subheader("📋 Dataset Sample")
    st.dataframe(df.head(10))
    
    # Basic statistics
    st.subheader("📈 Basic Statistics")
    st.dataframe(df.describe())
    
    # Missing values
    st.subheader("❓ Missing Values")
    missing_data = df.isnull().sum()
    if missing_data.sum() == 0:
        st.success("✅ No missing values found in the dataset!")
    else:
        st.dataframe(missing_data[missing_data > 0])

def show_data_analysis(df, data):
    """Display data analysis with visualizations"""
    st.header("📊 Data Analysis & Visualizations")
    
    # Grade distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Grade Distribution")
        fig_hist = px.histogram(df, x='G3', nbins=20, title='Final Grade (G3) Distribution',
                               color_discrete_sequence=['#1f77b4'])
        fig_hist.update_layout(xaxis_title="Final Grade", yaxis_title="Number of Students")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Pass/Fail Distribution")
        pass_fail_counts = data['Pass'].value_counts()
        fig_pie = px.pie(values=pass_fail_counts.values, names=['Fail', 'Pass'],
                        title='Pass/Fail Distribution', color_discrete_sequence=['#ff7f7f', '#7fbf7f'])
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Gender analysis
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("👥 Performance by Gender")
        gender_performance = df.groupby('sex')['G3'].mean().reset_index()
        fig_bar = px.bar(gender_performance, x='sex', y='G3', 
                        title='Average Grade by Gender',
                        color_discrete_sequence=['#ff7f0e'])
        fig_bar.update_layout(xaxis_title="Gender", yaxis_title="Average Grade")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col4:
        st.subheader("📚 Study Time vs Performance")
        fig_scatter = px.scatter(df, x='studytime', y='G3', color='sex',
                               title='Study Time vs Final Grade',
                               labels={'studytime': 'Study Time', 'G3': 'Final Grade'})
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Additional analysis
    st.subheader("🔍 Additional Insights")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("🏠 Family Support Impact")
        family_support = df.groupby('famsup')['G3'].mean().reset_index()
        fig_fam = px.bar(family_support, x='famsup', y='G3',
                        title='Average Grade by Family Support',
                        color_discrete_sequence=['#2ca02c'])
        st.plotly_chart(fig_fam, use_container_width=True)
    
    with col6:
        st.subheader("🌐 Internet Access Impact")
        internet_impact = df.groupby('internet')['G3'].mean().reset_index()
        fig_internet = px.bar(internet_impact, x='internet', y='G3',
                             title='Average Grade by Internet Access',
                             color_discrete_sequence=['#d62728'])
        st.plotly_chart(fig_internet, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Feature Correlations")
    numeric_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 
                   'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
    corr_matrix = df[numeric_cols].corr()
    
    fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title='Feature Correlation Heatmap',
                           color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_heatmap, use_container_width=True)

def show_model_training(data):
    """Display model training results"""
    st.header("🤖 Model Training & Evaluation")
    
    # Train the model
    with st.spinner("Training the model..."):
        model, X_train, X_test, y_train, y_test, y_pred, accuracy, feature_cols = train_model(data)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("Training Samples", len(X_train))
    with col3:
        st.metric("Test Samples", len(X_test))
    
    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                      labels=dict(x="Predicted", y="Actual"),
                      x=['Fail', 'Pass'], y=['Fail', 'Pass'],
                      color_continuous_scale='Blues',
                      title='Confusion Matrix')
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Feature Importance
    st.subheader("🎯 Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig_importance = px.bar(feature_importance.tail(15), x='importance', y='feature',
                           orientation='h', title='Top 15 Most Important Features',
                           color_discrete_sequence=['#9467bd'])
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Classification Report
    st.subheader("📋 Detailed Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    # Store model in session state for prediction
    st.session_state['model'] = model
    st.session_state['feature_cols'] = feature_cols

def show_prediction_interface(data, label_encoders):
    """Display prediction interface"""
    st.header("🔮 Student Performance Prediction")
    
    # Check if model is trained
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train the model first by visiting the 'Model Training' page.")
        return
    
    model = st.session_state['model']
    feature_cols = st.session_state['feature_cols']
    
    st.subheader("📝 Enter Student Information")
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Gender", ["F", "M"])
        age = st.slider("Age", 15, 22, 17)
        address = st.selectbox("Address", ["U", "R"])
        famsize = st.selectbox("Family Size", ["LE3", "GT3"])
        
    with col2:
        Pstatus = st.selectbox("Parent Status", ["T", "A"])
        Medu = st.slider("Mother's Education", 0, 4, 2)
        Fedu = st.slider("Father's Education", 0, 4, 2)
        traveltime = st.slider("Travel Time", 1, 4, 2)
        studytime = st.slider("Study Time", 1, 4, 2)
        
    with col3:
        failures = st.slider("Past Failures", 0, 3, 0)
        schoolsup = st.selectbox("School Support", ["yes", "no"])
        famsup = st.selectbox("Family Support", ["yes", "no"])
        higher = st.selectbox("Higher Education", ["yes", "no"])
        internet = st.selectbox("Internet Access", ["yes", "no"])
    
    # Additional features
    with st.expander("📋 Additional Information (Optional)"):
        col4, col5, col6 = st.columns(3)
        
        with col4:
            Mjob = st.selectbox("Mother's Job", ["at_home", "health", "other", "services", "teacher"])
            Fjob = st.selectbox("Father's Job", ["at_home", "health", "other", "services", "teacher"])
            reason = st.selectbox("Reason to Choose School", ["course", "other", "home", "reputation"])
            
        with col5:
            guardian = st.selectbox("Guardian", ["mother", "father", "other"])
            paid = st.selectbox("Extra Paid Classes", ["yes", "no"])
            activities = st.selectbox("Extra Activities", ["yes", "no"])
            
        with col6:
            nursery = st.selectbox("Nursery School", ["yes", "no"])
            romantic = st.selectbox("Romantic Relationship", ["yes", "no"])
            famrel = st.slider("Family Relationship", 1, 5, 4)
    
    # More features
    with st.expander("🎯 Lifestyle & Health"):
        col7, col8 = st.columns(2)
        
        with col7:
            freetime = st.slider("Free Time", 1, 5, 3)
            goout = st.slider("Going Out", 1, 5, 3)
            Dalc = st.slider("Workday Alcohol", 1, 5, 1)
            
        with col8:
            Walc = st.slider("Weekend Alcohol", 1, 5, 1)
            health = st.slider("Health Status", 1, 5, 3)
            absences = st.slider("Number of Absences", 0, 93, 5)
    
    # Prediction button
    if st.button("🔮 Predict Performance", type="primary"):
        # Prepare input data
        input_data = {
            'school': school, 'sex': sex, 'age': age, 'address': address, 'famsize': famsize,
            'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu, 'Mjob': Mjob, 'Fjob': Fjob,
            'reason': reason, 'guardian': guardian, 'traveltime': traveltime, 'studytime': studytime,
            'failures': failures, 'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
            'activities': activities, 'nursery': nursery, 'higher': higher, 'internet': internet,
            'romantic': romantic, 'famrel': famrel, 'freetime': freetime, 'goout': goout,
            'Dalc': Dalc, 'Walc': Walc, 'health': health, 'absences': absences
        }
        
        # Encode categorical variables
        for col, value in input_data.items():
            if col in label_encoders and isinstance(value, str):
                try:
                    input_data[col] = label_encoders[col].transform([value])[0]
                except ValueError:
                    # Handle unseen categories
                    input_data[col] = 0
        
        # Create DataFrame with the same structure as training data
        input_df = pd.DataFrame([input_data])
        
        # Ensure all feature columns are present
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Select only the features used in training
        input_df = input_df[feature_cols]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Display results
        st.subheader("🎯 Prediction Results")
        
        if prediction == 1:
            st.success("✅ **The student is likely to PASS!**")
            st.balloons()
        else:
            st.error("❌ **The student is at risk of FAILING.**")
        
        # Display probabilities
        col_prob1, col_prob2 = st.columns(2)
        
        with col_prob1:
            st.metric("Probability of Passing", f"{probability[1]:.1%}")
        with col_prob2:
            st.metric("Probability of Failing", f"{probability[0]:.1%}")
        
        # Probability visualization
        prob_data = pd.DataFrame({
            'Outcome': ['Fail', 'Pass'],
            'Probability': [probability[0], probability[1]]
        })
        
        fig_prob = px.bar(prob_data, x='Outcome', y='Probability',
                         title='Prediction Probabilities',
                         color='Outcome',
                         color_discrete_map={'Fail': '#ff7f7f', 'Pass': '#7fbf7f'})
        fig_prob.update_layout(yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_prob, use_container_width=True)

if __name__ == "__main__":
    main()
