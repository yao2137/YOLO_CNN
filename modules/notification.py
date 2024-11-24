import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

class Notification:
    def __init__(self, config, max_retries=3, base_retry_delay=2, max_threads=5):
        """
        Initializes the Notification module with async support.

        Args:
            config (dict): Configuration dictionary for notifications.
            max_retries (int): Maximum number of retries for failed notifications.
            base_retry_delay (int): Initial delay (in seconds) for retry, doubled on each failure.
            max_threads (int): Maximum number of threads for parallel notifications.
        """
        self.config = config
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.executor = ThreadPoolExecutor(max_threads)

        # Logger setup
        self.logger = logging.getLogger("Notification")
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Priority levels
        self.priority_levels = {"fall": 1, "run": 2, "static": 3, "crawl": 4}

    def _retry_logic(self, send_function, *args, **kwargs):
        """
        Implements retry logic with exponential backoff.

        Args:
            send_function (function): The notification function to retry.
            *args: Positional arguments for the send function.
            **kwargs: Keyword arguments for the send function.
        """
        delay = self.base_retry_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                send_function(*args, **kwargs)
                return "success"  # Exit if successful
            except Exception as e:
                self.logger.error(f"Attempt {attempt} failed: {e}")
                if attempt == self.max_retries:
                    self.logger.error("Max retries reached. Notification failed.")
                    return "failed"
                else:
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff

    def send_sms(self, message):
        """Sends an SMS notification via Twilio."""
        def send():
            twilio_config = self.config.get("twilio")
            client = Client(twilio_config["account_sid"], twilio_config["auth_token"])
            client.messages.create(
                body=message,
                from_=twilio_config["from_number"],
                to=twilio_config["to_number"]
            )
            self.logger.info("SMS sent successfully.")

        return self._retry_logic(send)

    def send_email(self, subject, message):
        """Sends an email notification via SMTP."""
        def send():
            email_config = self.config.get("email")
            msg = MIMEMultipart()
            msg["From"] = email_config["from_email"]
            msg["To"] = email_config["to_email"]
            msg["Subject"] = subject
            msg.attach(MIMEText(message, "plain"))

            with smtplib.SMTP(email_config["smtp_server"], email_config["port"]) as server:
                server.starttls()
                server.login(email_config["username"], email_config["password"])
                server.send_message(msg)
                self.logger.info("Email sent successfully.")

        return self._retry_logic(send)

    def send_push_notification(self, title, message):
        """Sends a push notification via Firebase Cloud Messaging."""
        def send():
            firebase_config = self.config.get("firebase")
            headers = {
                "Authorization": f"key={firebase_config['server_key']}",
                "Content-Type": "application/json"
            }
            payload = {
                "to": f"/topics/{firebase_config['topic']}",
                "notification": {
                    "title": title,
                    "body": message
                }
            }
            response = requests.post("https://fcm.googleapis.com/fcm/send", headers=headers, json=payload)
            if response.status_code != 200:
                raise Exception(f"Firebase notification failed: {response.text}")
            self.logger.info("Push notification sent successfully.")

        return self._retry_logic(send)

    def async_notify(self, behavior, track_id, additional_info=None):
        """
        Asynchronously sends notifications for a detected behavior.

        Args:
            behavior (str): Detected behavior.
            track_id (int): Target ID associated with the behavior.
            additional_info (dict): Optional additional data for the notification.
        """
        self.executor.submit(self.notify, behavior, track_id, additional_info)

    def notify(self, behavior, track_id, additional_info=None):
        """
        Sends notifications based on behavior priority.

        Args:
            behavior (str): Detected behavior.
            track_id (int): Target ID associated with the behavior.
            additional_info (dict): Optional additional data for the notification.
        """
        priority = self.priority_levels.get(behavior, 99)  # Default to lowest priority
        message_template = "[Priority {priority}] Alert: {behavior} detected!\nTrack ID: {track_id}\nDetails: {details}"
        details = additional_info if additional_info else "N/A"
        message = message_template.format(priority=priority, behavior=behavior, track_id=track_id, details=details)

        self.logger.info(f"Sending notifications for behavior: {behavior} (Priority {priority})")

        # High-priority notifications use all channels
        if priority == 1:
            self.send_sms(message)
            self.send_email("Critical Alert", message)
            self.send_push_notification("Critical Alert", message)
        # Medium-priority notifications use only push and email
        elif priority <= 3:
            self.send_email("Behavior Alert", message)
            self.send_push_notification("Behavior Alert", message)
        # Low-priority notifications only use email
        else:
            self.send_email("Behavior Notification", message)