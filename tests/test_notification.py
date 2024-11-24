import unittest
from modules.notification import Notification

class TestNotificationWithRetryAndPriority(unittest.TestCase):
    def setUp(self):
        # Mock configuration
        self.config = {
            "twilio": {
                "account_sid": "test_sid",
                "auth_token": "test_token",
                "from_number": "+1234567890",
                "to_number": "+0987654321"
            },
            "email": {
                "smtp_server": "smtp.test.com",
                "port": 587,
                "username": "test_user",
                "password": "test_pass",
                "from_email": "test_email@test.com",
                "to_email": "recipient@test.com"
            },
            "firebase": {
                "server_key": "test_server_key",
                "topic": "test_topic"
            }
        }
        self.notification = Notification(self.config)

    def test_retry_logic(self):
        """Test retry logic with exponential backoff."""
        self.notification.send_sms("Test retry logic message")
        # Verify retry behavior in logs or by observing service call frequency

    def test_priority_handling(self):
        """Test priority-based notification routing."""
        self.notification.notify("fall", 1, {"details": "Critical event"})
        self.notification.notify("run", 2, {"details": "Moderate event"})
        self.notification.notify("crawl", 3, {"details": "Low-priority event"})
        # Verify channels used for each priority in logs

if __name__ == "__main__":
    unittest.main()
