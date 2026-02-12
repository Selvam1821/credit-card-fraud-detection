import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import io

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

# Title
st.title("🚀 Credit Card Fraud Detection")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Adjust settings for fraud detection.")
    threshold = st.slider("Fraud Detection Threshold", 0.0, 1.0, 0.5)
    st.write(f"Threshold set to: {threshold}")
    if st.button("Clear History"):
        st.session_state.transaction_history = []
        st.success("Transaction history cleared!")

# Main content with columns
col1, col2 = st.columns([2, 1])

with col1:
    # Transaction details input
    st.subheader("🔍 Enter Transaction Details")
    transaction_time = st.number_input("Transaction Time (seconds)", min_value=0, step=1, help="Enter the time in seconds.")
    transaction_amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01, help="Enter the transaction amount.")
    email = st.text_input("Your Email", help="Enter your email address.")

    # Simple rule-based fraud detection (replace with a real ML model)
    def detect_fraud(amount, time, threshold):
        # Basic rule: flag as fraud if amount is high or time is unusually short
        score = (amount / 1000) + (1 / (time + 1))  # Arbitrary formula for demo
        return "Fraudulent" if score > threshold else "Safe"

    # Check for fraud
    if st.button("Check for Fraud"):
        if not email:
            st.warning("Please enter an email address.")
        else:
            result = detect_fraud(transaction_amount, transaction_time, threshold)
            if result == "Fraudulent":
                st.error("⚠️ This transaction is flagged as FRAUDULENT!")
            else:
                st.success("✅ This transaction is SAFE!")
            
            # Add to transaction history
            st.session_state.transaction_history.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Email": email,
                "Amount": transaction_amount,
                "Time (s)": transaction_time,
                "Result": result
            })

with col2:
    # Fraud detection statistics
    st.subheader("📊 Fraud Detection Statistics")
    
    # Calculate stats from transaction history
    if st.session_state.transaction_history:
        df_history = pd.DataFrame(st.session_state.transaction_history)
        fraud_counts = df_history['Result'].value_counts().to_dict()
        fraudulent = fraud_counts.get("Fraudulent", 0)
        safe = fraud_counts.get("Safe", 0)
    else:
        fraudulent, safe = 30, 70  # Default values if no history

    data = {'Category': ['Fraudulent Transactions', 'Safe Transactions'], 'Count': [fraudulent, safe]}
    df = pd.DataFrame(data)
    
    # Interactive pie chart
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

# Transaction history section
st.subheader("📜 Transaction History")
if st.session_state.transaction_history:
    df_history = pd.DataFrame(st.session_state.transaction_history)
    st.dataframe(df_history, use_container_width=True)
    
    # Export to CSV
    csv = df_history.to_csv(index=False)
    st.download_button(
        label="📥 Download History as CSV",
        data=csv,
        file_name="transaction_history.csv",
        mime="text/csv",
    )
else:
    st.info("No transactions yet. Perform a fraud check to see the history.")

# Footer
