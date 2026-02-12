
import smtplib

EMAIL_ADDRESS = "gatewaykesavan@gmail.com"
EMAIL_PASSWORD = "gvltvwrpgyticqns"

with smtplib.SMTP("smtp.gmail.com", 587) as server:
    server.starttls()
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    server.sendmail(
        EMAIL_ADDRESS,
        "gatewaykesavan@gmail.com",
        "Subject: Test Email\n\nHello, this is a test."
    )

print("✅ Email sent successfully!")
