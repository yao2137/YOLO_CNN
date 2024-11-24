Home Elderly Activity and Safety Monitoring System

Introduction

This project aims to monitor the activities of elderly individuals at home to ensure their safety and well-being. It primarily focuses on detecting dangerous events, such as falls, and notifying family members immediately via SMS or email alerts. Additionally, it tracks regular daily activities, such as eating, sleeping, sitting, and moving, to help caregivers better understand the elderly’s routines.

By leveraging state-of-the-art computer vision algorithms, pre-trained models, and configurable modules, this system provides reliable real-time monitoring with minimal user intervention.

Core Features

	1.	Danger Detection and Alerts:
	•	Detects falls and other critical behaviors in real-time.
	•	Sends SMS or email notifications to family members when a dangerous event occurs.
	2.	Daily Activity Monitoring:
	•	Tracks activities like sitting, lying down, eating, running, and crawling.
	•	Logs behaviors with timestamps for review and analysis.
	3.	Real-Time Performance:
	•	Uses advanced object detection and behavior classification algorithms.
	•	Processes live video streams from home-installed cameras.
	4.	Danger Zone Monitoring:
	•	Configures specific zones (e.g., staircases or kitchens) to monitor entry and exit events.
	5.	Behavior Reports:
	•	Maintains a timeline of activities for each tracked individual.
	•	Saves critical event snapshots for review.
	6.	Environmental Adaptation:
	•	Dynamically adjusts detection thresholds based on lighting and noise conditions.

Technologies Used

Datasets

	1.	Kinetics-400 Dataset:
	•	Used for pre-training the behavior classification model.
	•	Contains diverse human activity videos (e.g., eating, sitting, walking).
	2.	Custom Dataset:
	•	Includes videos of elderly individuals performing specific activities, such as sitting, crawling, or falling.

Models

	1.	YOLOv5:
	•	Pre-trained object detection model for identifying people and objects in frames.
	2.	ResNet18:
	•	Fine-tuned for classifying behaviors like sitting, eating, lying, and crawling.

Frameworks

	1.	PyTorch:
	•	For training and inference of deep learning models.
	2.	OpenCV:
	•	For video frame processing and visualization.
	3.	Shapely:
	•	For defining and detecting danger zones.

Other Tools

	•	Twilio API: For sending SMS alerts.
	•	Email Libraries: For sending email notifications.

Project Configuration

Configuration File: config/config.yaml

The configuration file defines thresholds, paths, and danger zones:

behavior:
  fall_threshold: 0.4
  static_threshold: 5
  speed_threshold: 2.0
  run_threshold: 4.0
  crawl_ratio: 0.5
models:
  detection_model_path: "models/yolov5/best.pt"
  classifier_model_path: "models/behavior/behavior_classifier.pt"
notification:
  sms:
    twilio_sid: "your_twilio_sid"
    twilio_token: "your_twilio_token"
    from_number: "+1234567890"
    to_number: "+0987654321"
  email:
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    sender_email: "youremail@gmail.com"
    sender_password: "yourpassword"
    recipient_email: "recipientemail@gmail.com"
zones:
  danger_zones:
    - [[100, 100], [200, 100], [200, 200], [100, 200]]

Setup Instructions

Prerequisites

	•	Python 3.8+
	•	A GPU with CUDA support (optional, for real-time performance)

Installation

	1.	Clone the repository:

git clone https://github.com/your-repo/behavior-recognition-system.git
cd behavior-recognition-system


	2.	Install dependencies:

pip install -r requirements.txt


	3.	Download pre-trained models:
	•	Place the YOLOv5 model (best.pt) in models/yolov5/.
	•	Place the behavior classifier (behavior_classifier.pt) in models/behavior/.

Usage

Running the System

	1.	Start the behavior recognition system:

python main.py


	2.	The system processes video frames from your webcam or configured RTSP stream.
	3.	When a dangerous behavior (e.g., falling) is detected:
	•	An SMS or email is sent to the configured recipients.
	•	Snapshots of the event are saved in data/outputs/.

Real-Time Monitoring

	•	Visualizes bounding boxes, activity labels, and danger zone entries on video frames.

Extending the System

Adding New Behaviors

	1.	Collect labeled data for the new behavior.
	2.	Fine-tune the ResNet18 model using the labeled data.
	3.	Update class_names in config.yaml to include the new behavior.

Adding New Danger Zones

	1.	Define the new danger zone as a polygon.
	2.	Update the danger_zones field in config.yaml.

Customizing Notifications

	1.	Edit the notification section in config.yaml to configure SMS and email alerts.
	2.	Use the notification.py module to add new notification channels (e.g., push notifications).

Testing and Validation

Unit Tests

Run unit tests to ensure the system is working as expected:

python -m unittest discover -s tests -p "test_*.py"

Validation

	•	Test the system with various real-life scenarios:
	•	Simulate falls, fast movements, and crawling.
	•	Test activity classification for sitting, lying, and eating.

Future Enhancements

	1.	Behavior Analytics Dashboard:
	•	Develop a web-based interface to display behavior logs and reports.
	2.	Improved Behavior Classification:
	•	Add support for more complex behaviors like group interactions.
	3.	Multi-Camera Support:
	•	Extend to handle video streams from multiple cameras simultaneously.
	4.	Cloud Integration:
	•	Store behavior logs and snapshots in the cloud for remote access.

FAQs

Q: How does the system detect dangerous events?

The system uses YOLOv5 to detect human targets and a ResNet18 classifier to recognize behaviors like falls. It dynamically adjusts detection thresholds for reliability.

Q: What happens when a fall is detected?

When a fall is detected:
	1.	An SMS or email is sent to the family member.
	2.	A snapshot of the event is saved for later review.

Q: Can I use a different camera?

Yes, the system supports both USB webcams and RTSP streams. Update the camera source in main.py.

License

This project is licensed under the MIT License. See the LICENSE file for details.

For support or further inquiries, feel free to open an issue or contact Your Name.