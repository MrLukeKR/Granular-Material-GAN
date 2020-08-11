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

time_fmt = "%m/%d/%Y, %H:%M:%S"


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


def send_experiment_success(experiment_id, start_time, end_time, g_loss, d_loss):
    total_time = (end_time - start_time)

    email_notification = """Hi there,

                                Experiment %s has successfully finished! 

                                Started: %s
                                Ended: %s
                                Time taken: %s

                                -- Generator Metrics --
                                Final Generator Loss: %s
                                Final Generator MSE: %s
                                -----------------------

                                -- Discriminator Metrics --
                                Final Discriminator Loss: %s
                                Final Discriminator Accuracy: %s
                                ---------------------------

                                ~ Automated Notification System
                             """ % (str(experiment_id),
                                    start_time.strftime(time_fmt), end_time.strftime(time_fmt), str(total_time),
                                    str(g_loss[0]), str(g_loss[1]),
                                    str(d_loss[0]), str(d_loss[1]))

    send_email(email_notification, "[Success Notification] Experiment %s Completed" % str(experiment_id))
