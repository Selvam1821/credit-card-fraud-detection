import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import io
import random

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

# Generate some sample data for analytics if not present
if 'sample_data' not in st.session_state:
    # Generate dates for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = [start_date + timedelta(days=i) for i in range(31)]
    
    # Generate random transaction counts and amounts
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

elif page == "Analytics Dashboard":
    st.title("📈 Fraud Detection Analytics")
    
    # Create a DataFrame from sample data
    df_analytics = pd.DataFrame({
        'Date': st.session_state.sample_data['date'],
        'Transactions': st.session_state.sample_data['transaction_count'],
        'Frauds': st.session_state.sample_data['fraud_count'],
        'Amount': st.session_state.sample_data['transaction_amount']
    })
    
    # Time period selector
    time_period = st.selectbox(
        "Select Time Period",
        ["Last 7 Days", "Last 14 Days", "Last 30 Days"]
    )
    
    # Filter data based on selected time period
    if time_period == "Last 7 Days":
        df_filtered = df_analytics.tail(7)
    elif time_period == "Last 14 Days":
        df_filtered = df_analytics.tail(14)
    else:
        df_filtered = df_analytics
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily Transaction vs Fraud Count")
        # Line chart with two y-axes
        fig = go.Figure()
        
        # Add transactions line
        fig.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['Transactions'],
            name="Total Transactions",
            line=dict(color="#3366cc", width=3)
        ))
        
        # Add fraud line
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
            xaxis=dict(
                title="Date",
                showgrid=False,
                tickformat="%d %b"
            ),
            yaxis=dict(
                title="Count",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.2)"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Fraud Percentage Over Time")
        
        # Calculate fraud percentage
        df_filtered['Fraud_Percentage'] = (df_filtered['Frauds'] / df_filtered['Transactions']) * 100
        
        # Area chart for fraud percentage
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
            xaxis=dict(
                title="Date",
                showgrid=False,
                tickformat="%d %b"
            ),
            yaxis=dict(
                title="Fraud Percentage (%)",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.2)",
                range=[0, max(df_filtered['Fraud_Percentage']) * 1.1]
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Third row with a single wide chart
    st.subheader("Transaction Amount Distribution")
    
    # Transaction amount histogram with KDE
    fig = px.histogram(
        df_filtered,
        x="Amount",
        nbins=20,
        opacity=0.7,
        color_discrete_sequence=["#66cc99"]
    )
    
    # Add KDE line
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
        xaxis=dict(
            title="Transaction Amount",
            showgrid=False
        ),
        yaxis=dict(
            title="Frequency",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.2)"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table with key metrics
    st.subheader("Summary Statistics")
    
    # Calculate metrics
    total_transactions = df_filtered['Transactions'].sum()
    total_frauds = df_filtered['Frauds'].sum()
    fraud_rate = (total_frauds / total_transactions) * 100 if total_transactions > 0 else 0
    avg_transaction = df_filtered['Amount'].mean()
    
    # Display metrics in columns
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