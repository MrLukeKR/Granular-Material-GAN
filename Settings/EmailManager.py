import smtplib
import ssl
import socket

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from Settings import SettingsManager as sm

user = None
server = None
port = None
password = None
default_recipient = None

context = ssl.create_default_context()

time_fmt = "%m/%d/%Y, %H:%M:%S"


def initialise():
    global context, server, port, user, password, default_recipient

    user = sm.auth.get("EMAIL_USER")
    server = sm.auth.get("EMAIL_HOST")
    port = sm.auth.get("EMAIL_PORT")
    password = sm.auth.get("EMAIL_PASS")
    default_recipient = sm.get_setting("EMAIL_NOTIFICATION_RECIPIENT")


def send_email(message, subject="[Automated Alert] Asphalt GAN Notification", recipient=None):
    global context, server, port, user, password, default_recipient

    email = MIMEMultipart("alternative")
    email["Subject"] = subject
    email["From"] = user
    email["To"] = recipient if recipient is not None else default_recipient

    plain = MIMEText(message, "plain")

    email.attach(plain)

    email_server = smtplib.SMTP(host=server, port=port)

    email_server.login(user, password)
    email_server.send_message(email)


def send_results_generation_success(experiment_id, start_time, end_time):
    total_time = (end_time - start_time)

    email_notification = """Hi there,

                            Results generation for %s has successfully finished on %s! 

                            Started: %s
                            Ended: %s
                            Time taken: %s

                            ~ Automated Notification System
                            """ % (str(experiment_id), socket.gethostname(),
                                   start_time.strftime(time_fmt), end_time.strftime(time_fmt), str(total_time))

    send_email(email_notification, "[Success Notification] Results for %s Completed" % str(experiment_id))


def send_experiment_success(experiment_id, start_time, end_time, g_loss, d_loss):
    total_time = (end_time - start_time)

    email_notification = """Hi there,

                            %s has successfully finished on %s! 

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
                            """ % (str(experiment_id), socket.gethostname(),
                                   start_time.strftime(time_fmt), end_time.strftime(time_fmt), str(total_time),
                                   str(g_loss[0][-1]), str(g_loss[1][-1]),
                                   str(d_loss[0][-1]), str(d_loss[1][-1] * 100) + '%')

    send_email(email_notification, "[Success Notification] GAN Training for %s Completed" % str(experiment_id))
