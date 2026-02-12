import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import io
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Custom CSS for a polished look
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

# Initialize session state for transaction history
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []

# Hardcoded email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "selvamsssankar@gmail.com"  # Replace with your email
SENDER_PASSWORD = "xwocumitdbvfhlar" # Replace with your app password
ADMIN_EMAIL = "selvamsssankar@gmail.com"

# Function to send fraud alert emails
def send_alert(user_email, amount, time):
    try:
        admin_email = ADMIN_EMAIL
        recipients = [user_email, admin_email]

        print(f"📧 Sending fraud alert to: {recipients}")  # Debug log

        # Create message for plain text email
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = ", ".join(recipients)  # Join recipients for SMTP
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

        # Connect to SMTP server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()

        print("✅ Email sent successfully!")
        return True
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return False

# Generate sample data for analytics
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

# Sidebar for navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Select Page", ["Fraud Detection", "Analytics Dashboard"])
    
    st.header("⚙️ Settings")
    st.write("Adjust settings for fraud detection.")
    threshold = st.slider("Fraud Detection Threshold", 0.0, 1.0, 0.5)
    st.write(f"Threshold set to: {threshold}")
    if st.button("Clear History"):
        st.session_state.transaction_history = []
        st.success("Transaction history cleared!")

# Main page content
if page == "Fraud Detection":
    st.title("🚀 Credit Card Fraud Detection")
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔍 Enter Transaction Details")
        transaction_time = st.number_input("Transaction Time (seconds)", min_value=0, step=1, help="Enter the time in seconds.")
        transaction_amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01, help="Enter the transaction amount.")
        email = st.text_input("Your Email", help="Enter your email address.")

        def detect_fraud(amount, time):
            # Flag as fraudulent if amount > 10000 OR time > 200 seconds
            if amount > 10000 or time > 200:
                return "Fraudulent"
            return "Safe"

        if st.button("Check for Fraud"):
            if not email:
                st.warning("Please enter an email address.")
            elif not transaction_amount or transaction_time < 0:
                st.warning("Please enter valid transaction details.")
            else:
                result = detect_fraud(transaction_amount, transaction_time)
                email_sent = False
                
                if result == "Fraudulent":
                    st.error("⚠️ This transaction is flagged as FRAUDULENT!")
                    email_sent = send_alert(email, transaction_amount, transaction_time)
                    if email_sent:
                        st.success("📧 Fraud alert email sent to user and admin!")
                    else:
                        st.warning("⚠️ Failed to send fraud alert email.")
                else:
                    st.success("✅ This transaction is SAFE!")
                
                st.session_state.transaction_history.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Email": email,
                    "Amount": transaction_amount,
                    "Time (s)": transaction_time,
                    "Result": result,
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
        
        fig = px.pie(df, values='Count', names='Category', 
                     color='Category', 
                     color_discrete_map={'Fraudulent Transactions': '#ff4b5c', 'Safe Transactions': '#66cc99'},
                     title="Transaction Breakdown")
        fig.update_traces(textinfo='percent+label', pull=[0.1, 0], marker=dict(line=dict(color='#000000', width=2)))
        fig.update_layout(
            title_font_size=20,
            title_font_color="#ffeb3b",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📜 Transaction History")
    if st.session_state.transaction_history:
        df_history = pd.DataFrame(st.session_state.transaction_history)
        st.dataframe(df_history, use_container_width=True)
        
        csv = df_history.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name="transaction_history.csv",
            mime="text/csv",
        )
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
        fig.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['Transactions'],
            name="Total Transactions",
            line=dict(color="#3366cc", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['Frauds'],
            name="Fraudulent Transactions",
            line=dict(color="#ff4b5c", width=3)
        ))
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.05)",
            font=dict(color="white"),
            xaxis=dict(title="Date", showgrid=False, tickformat="%d %b"),
            yaxis=dict(title="Count", showgrid=True, gridcolor="rgba(255,255,255,0.2)")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Fraud Percentage Over Time")
        df_filtered['Fraud_Percentage'] = (df_filtered['Frauds'] / df_filtered['Transactions']) * 100
        fig = px.area(
            df_filtered, 
            x='Date', 
            y='Fraud_Percentage',
            title=None,
            color_discrete_sequence=["#ff6f7d"]
        )
        fig.update_layout(
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.05)",
            font=dict(color="white"),
            xaxis=dict(title="Date", showgrid=False, tickformat="%d %b"),
            yaxis=dict(
                title="Fraud Percentage (%)",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.2)",
                range=[0, max(df_filtered['Fraud_Percentage']) * 1.1]
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Transaction Amount Distribution")
    fig = px.histogram(
        df_filtered,
        x="Amount",
        nbins=20,
        opacity=0.7,
        color_discrete_sequence=["#66cc99"]
    )
    kde_values = np.histogram(df_filtered["Amount"], bins=20)[0]
    kde_x = np.histogram(df_filtered["Amount"], bins=20)[1]
    kde_x = [(kde_x[i] + kde_x[i+1])/2 for i in range(len(kde_x)-1)]
    fig.add_trace(
        go.Scatter(
            x=kde_x,
            y=kde_values,
            mode='lines',
            name='Distribution',
            line=dict(color="#ffeb3b", width=3)
        )
    )
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        font=dict(color="white"),
        xaxis=dict(title="Transaction Amount", showgrid=False),
        yaxis=dict(title="Frequency", showgrid=True, gridcolor="rgba(255,255,255,0.2)")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Summary Statistics")
    total_transactions = df_filtered['Transactions'].sum()
    total_frauds = df_filtered['Frauds'].sum()
    fraud_rate = (total_frauds / total_transactions) * 100 if total_transactions > 0 else 0
    avg_transaction = df_filtered['Amount'].mean()
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Total Transactions", f"{total_transactions:,}")
    with metric_col2:
        st.metric("Fraudulent Transactions", f"{total_frauds:,}")
    with metric_col3:
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    with metric_col4:
        st.metric("Avg Transaction Amount", f"${avg_transaction:.2f}")

# Footer
st.markdown("---")
st.markdown("© 2025 Credit Card Fraud Detection System")