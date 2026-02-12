import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #ff4b5c;
        color: white;
        border-radius: 10px;
        font-size: 18px;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff6f7d;
        transform: scale(1.05);
    }
    .stTextInput>div>input, .stNumberInput>div>input {
        background-color: #ffffff;
        color: #333;
        border-radius: 8px;
        border: 2px solid #ff4b5c;
        padding: 10px;
    }
    h1, h2, h3 {
        color: #ffeb3b;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .sidebar .sidebar-content {
        background: #2a5298;
        color: white;
    }
    .stDataFrame {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []
if 'sample_data' not in st.session_state:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = [start_date + timedelta(days=i) for i in range(31)]
    daily_counts = [random.randint(5, 50) for _ in range(31)]
    daily_frauds = [random.randint(0, max(1, int(count * 0.2))) for count in daily_counts]
    daily_amounts = [random.uniform(500, 5000) for _ in range(31)]
    st.session_state.sample_data = {
        'date': dates,
        'transaction_count': daily_counts,
        'fraud_count': daily_frauds,
        'transaction_amount': daily_amounts
    }

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "gatewaykesavan@gmail.com"  # Replace with your email
SENDER_PASSWORD = "gvltvwrpgyticqns"  # Replace with your app password
ADMIN_EMAIL = "gatewaykesavan@gmail.com"

# Load and train model
@st.cache_resource
def load_and_train_model():
    try:
        st.write("Loading dataset...")
        data = pd.read_csv('creditcard.csv')
        data.dropna(subset=['Class'], inplace=True)
        
        # Sample 10% of the data to speed up training
        data = data.sample(frac=0.1, random_state=42)
        
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
        
        # Apply SMOTE with sampling strategy to limit oversampling
        smote = SMOTE(random_state=42, sampling_strategy=0.5)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train Random Forest (optimized parameters)
        status_text.text("Training Random Forest...")
        model = RandomForestClassifier(random_state=42, n_estimators=30, max_depth=8, n_jobs=-1)
        model.fit(X_train_smote, y_train_smote)
        progress_bar.progress(50)
        
        # Save model and scaler
        joblib.dump(model, "fraud_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        
        # Evaluate Random Forest
        status_text.text("Evaluating model...")
        evaluation_results = {}
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_pred)
        evaluation_results['Random Forest'] = {'confusion_matrix': cm, 'report': report, 'roc_auc': roc_auc}
        
        status_text.text("Model training and evaluation complete!")
        progress_bar.progress(100)
        
        return model, scaler, evaluation_results, X.columns
    except Exception as e:
        st.error(f"❌ Error loading/training model: {e}")
        return None, None, None, None

model, scaler, evaluation_results, feature_names = load_and_train_model()

# Send fraud alert emails
def send_alert(user_email, amount, time):
    try:
        recipients = [user_email, ADMIN_EMAIL]
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = ", ".join(recipients)
        msg['Subject'] = f"🚨 Fraud Alert! Suspicious Transaction of ${amount}"
        msg_body = (
            f"Dear Customer,\n\n"
            f"We have detected a suspicious transaction:\n"
            f"🔹 Amount: ${amount:.2f}\n"
            f"🔹 Time: {time} seconds\n"
            f"🔹 Status: 🚨 Potential Fraud\n\n"
            f"If this was not you, please contact your bank immediately!\n\n"
            f"Regards,\nFraud Detection Team"
        )
        msg.attach(MIMEText(msg_body, 'plain'))
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"❌ Error sending email: {e}")
        return False

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Select Page", ["Fraud Detection", "Analytics Dashboard", "Model Evaluation"])
    st.header("⚙️ Settings")
    st.write("Adjust settings for fraud detection.")
    threshold = st.slider("Fraud Detection Threshold", 0.0, 1.0, 0.5)
    if st.button("Clear History"):
        st.session_state.transaction_history = []
        st.success("Transaction history cleared!")

# Fraud detection function
def detect_fraud(amount, time, email):
    if model is None or scaler is None:
        return "Unknown", 0.0
    # Create a feature vector
    features = np.zeros((1, len(feature_names)))
    try:
        time_idx = feature_names.get_loc('Time')
        amount_idx = feature_names.get_loc('Amount')
        features[0, time_idx] = time
        features[0, amount_idx] = amount
    except:
        # Fallback: Use only Time and Amount
        features = np.array([[time, amount]])
    # Scale features
    features_scaled = scaler.transform(features)
    prob = model.predict_proba(features_scaled)[0, 1]
    prediction = 1 if prob >= threshold else 0
    return "Fraudulent" if prediction == 1 else "Safe", prob

# Main page content
if page == "Fraud Detection":
    st.title("🚀 Credit Card Fraud Detection")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Enter Transaction Details")
        transaction_time = st.number_input("Transaction Time (seconds)", min_value=0, step=1)
        transaction_amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
        email = st.text_input("Your Email")
        
        if st.button("Check for Fraud"):
            if not email:
                st.warning("Please enter an email address.")
            elif not transaction_amount or transaction_time < 0:
                st.warning("Please enter valid transaction details.")
            else:
                result, prob = detect_fraud(transaction_amount, transaction_time, email)
                email_sent = False
                if result == "Fraudulent":
                    st.error(f"⚠️ This transaction is flagged as FRAUDULENT! (Probability: {prob:.2f})")
                    email_sent = send_alert(email, transaction_amount, transaction_time)
                    if email_sent:
                        st.success("📧 Fraud alert email sent to user and admin!")
                    else:
                        st.warning("⚠️ Failed to send fraud alert email.")
                else:
                    st.success(f"✅ This transaction is SAFE! (Probability of fraud: {prob:.2f})")
                
                st.session_state.transaction_history.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Email": email,
                    "Amount": transaction_amount,
                    "Time (s)": transaction_time,
                    "Result": result,
                    "Fraud Probability": prob,
                    "Email Sent": email_sent
                })
    
    with col2:
        st.subheader("📊 Fraud Detection Statistics")
        if st.session_state.transaction_history:
            df_history = pd.DataFrame(st.session_state.transaction_history)
            fraud_counts = df_history['Result'].value_counts().to_dict()
            fraudulent = fraud_counts.get("Fraudulent", 0)
            safe = fraud_counts.get("Safe", 0)
        else:
            fraudulent, safe = 30, 70
        data = {'Category': ['Fraudulent Transactions', 'Safe Transactions'], 'Count': [fraudulent, safe]}
        df = pd.DataFrame(data)
        fig = px.pie(df, values='Count', names='Category', color='Category',
                     color_discrete_map={'Fraudulent Transactions': '#ff4b5c', 'Safe Transactions': '#66cc99'},
                     title="Transaction Breakdown")
        fig.update_traces(textinfo='percent+label', pull=[0.1, 0], marker=dict(line=dict(color='#000000', width=2)))
        fig.update_layout(title_font_size=20, title_font_color="#ffeb3b", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📜 Transaction History")
    if st.session_state.transaction_history:
        df_history = pd.DataFrame(st.session_state.transaction_history)
        st.dataframe(df_history, use_container_width=True)
        csv = df_history.to_csv(index=False)
        st.download_button(label="📥 Download History as CSV", data=csv, file_name="transaction_history.csv", mime="text/csv")
    else:
        st.info("No transactions yet.")

elif page == "Analytics Dashboard":
    st.title("📈 Fraud Detection Analytics")
    df_analytics = pd.DataFrame({
        'Date': st.session_state.sample_data['date'],
        'Transactions': st.session_state.sample_data['transaction_count'],
        'Frauds': st.session_state.sample_data['fraud_count'],
        'Amount': st.session_state.sample_data['transaction_amount']
    })
    
    time_period = st.selectbox("Select Time Period", ["Last 7 Days", "Last 14 Days", "Last 30 Days"])
    if time_period == "Last 7 Days":
        df_filtered = df_analytics.tail(7).copy()
    elif time_period == "Last 14 Days":
        df_filtered = df_analytics.tail(14).copy()
    else:
        df_filtered = df_analytics.copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily Transaction vs Fraud Count")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Transactions'], name="Total Transactions",
                                 line=dict(color="#3366cc", width=3)))
        fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Frauds'], name="Fraudulent Transactions",
                                 line=dict(color="#ff4b5c", width=3)))
        fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                          margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
                          font=dict(color="white"), xaxis=dict(title="Date", showgrid=False, tickformat="%d %b"),
                          yaxis=dict(title="Count", showgrid=True, gridcolor="rgba(255,255,255,0.2)"))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Fraud Percentage Over Time")
        df_filtered['Fraud_Percentage'] = (df_filtered['Frauds'] / df_filtered['Transactions']) * 100
        fig = px.area(df_filtered, x='Date', y='Fraud_Percentage', title=None, color_discrete_sequence=["#ff6f7d"])
        fig.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0.05)", font=dict(color="white"),
                          xaxis=dict(title="Date", showgrid=False, tickformat="%d %b"),
                          yaxis=dict(title="Fraud Percentage (%)", showgrid=True, gridcolor="rgba(255,255,255,0.2)",
                                     range=[0, max(df_filtered['Fraud_Percentage']) * 1.1]))
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Transaction Scatter Plot (Amount vs Time)")
    np.random.seed(42)
    scatter_data = pd.DataFrame({
        'Time': np.random.uniform(0, 172800, 1000),
        'Amount': np.random.uniform(0, 5000, 1000),
        'Fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    })
    fig = px.scatter(scatter_data, x='Time', y='Amount', color='Fraud', color_continuous_scale=['#66cc99', '#ff4b5c'],
                     labels={'Fraud': 'Transaction Type'}, title="Transactions: Amount vs Time")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)", font=dict(color="white"),
                      xaxis=dict(title="Time (seconds)", showgrid=False),
                      yaxis=dict(title="Amount ($)", showgrid=True, gridcolor="rgba(255,255,255,0.2)"))
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Feature Importance (Random Forest)")
    if model is not None:
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(10)
        fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', color='Importance',
                     color_continuous_scale='Reds', title="Top 10 Feature Importance")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)", font=dict(color="white"),
                          xaxis=dict(title="Importance", showgrid=False),
                          yaxis=dict(title="Feature", showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Evaluation":
    st.title("📊 Model Evaluation Results")
    if evaluation_results:
        for name, results in evaluation_results.items():
            st.subheader(f"{name}")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Classification Report**")
                report_df = pd.DataFrame(results['report']).transpose()
                st.dataframe(report_df)
                st.write(f"**ROC-AUC Score**: {results['roc_auc']:.4f}")
            
            with col2:
                st.write("**Confusion Matrix**")
                cm = results['confusion_matrix']
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted Safe', 'Predicted Fraud'],
                    y=['Actual Safe', 'Actual Fraud'],
                    text=cm,
                    texttemplate="%{text}",
                    colorscale='Blues',
                    showscale=False
                ))
                fig.update_layout(
                    title=f'Confusion Matrix for {name}',
                    xaxis_title="Predicted Label",
                    yaxis_title="True Label",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.05)",
                    font=dict(color="white"),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No evaluation results available.")

# Footer
st.markdown("---")
st.markdown("© 2025 Credit Card Fraud Detection System")