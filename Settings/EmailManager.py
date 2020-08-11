import smtplib
import ssl

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from Settings import SettingsManager as sm

user = sm.auth.get("EMAIL_USER")
server = sm.auth.get("EMAIL_HOST")
port = sm.auth.get("EMAIL_PORT")
password = sm.auth.get("EMAIL_PASS")

context = ssl.create_default_context()

default_recipient = sm.get_setting("EMAIL_NOTIFICATION_RECIPIENT")


def send_email(message, subject="[Automated Alert] Asphalt GAN Notification", recipient=default_recipient):
    global context, server, port, user

    email = MIMEMultipart("alternative")
    email["Subject"] = subject
    email["From"] = user
    email["To"] = recipient

    plain = MIMEText(message, "plain")

    email.attach(plain)

    with smtplib.SMTP_SSL(server, port, context=context) as server:
        server.login(user, password)

        server.sendmail(user, recipient, email.as_string())
