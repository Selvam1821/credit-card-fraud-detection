from flask import Flask, render_template, request, jsonify
from flask_mail import Mail, Message
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("fraud_model.pkl")

# Email Configuration (Stored in config.py)
app.config.from_pyfile("config.py")
mail = Mail(app)

# Function to send fraud alert emails
def send_alert(user_email, amount):
    try:
        admin_email = "gatewaykesavan@gmail.com"  # Fixed admin email
        recipients = [user_email, admin_email]  # Send to both user & admin

        print(f"📧 Sending fraud alert to: {recipients}")  # Debug log

        msg = Message(
            subject=f"🚨 Fraud Alert! Suspicious Transaction of ${amount}",
            sender=app.config["MAIL_USERNAME"],
            recipients=recipients
        )

        msg.body = (
            f"Dear Customer,\n\n"
            f"We have detected a suspicious transaction:\n"
            f"🔹 Amount: ${amount}\n"
            f"🔹 Status: 🚨 Potential Fraud\n\n"
            f"If this was not you, please contact your bank immediately!\n\n"
            f"Regards,\nFraud Detection Team"
        )

        mail.send(msg)
        print("✅ Email sent successfully!")

    except Exception as e:
        print(f"❌ Error sending email: {e}")  # Debugging info

# Home route - Serves the frontend UI
@app.route("/")
def home():
    return render_template("index.html")

# Fraud detection API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get JSON data
    amount = data.get("Amount")
    time = data.get("Time")
    email = data.get("Email")  # Get user email

    if not all([amount, time, email]):  # Check for missing values
        return jsonify({"Error": "Missing data"}), 400

    # Convert to NumPy array for model prediction
    features = np.array([[time, amount]])
    
    # Predict fraud (1: normal, -1: fraud)
    prediction = model.predict(features)[0]
    
    if prediction == -1:  # If fraud detected
        send_alert(email, amount)
        return jsonify({"Fraud": True, "Message": "🚨 Fraud detected! Email alert sent."})
    
    return jsonify({"Fraud": False, "Message": "✅ Transaction is safe."})

if __name__ == "__main__":
    app.run(debug=True)
