#h1 Home Elderly Activity and Safety Monitoring System

Introduction

This project aims to monitor the activities of elderly individuals at home to ensure their safety and well-being. It primarily focuses on detecting dangerous events, such as falls, and notifying family members immediately via SMS or email alerts. Additionally, it tracks regular daily activities, such as eating, sleeping, sitting, and moving, to help caregivers better understand the elderly’s routines.

By leveraging state-of-the-art computer vision algorithms, pre-trained models, and configurable modules, this system provides reliable real-time monitoring with minimal user intervention.

Core Features

	1.	Danger Detection and Alerts:
		Detects falls and other critical behaviors in real-time.
		Sends SMS or email notifications to family members when a dangerous event occurs.
	2.	Daily Activity Monitoring:
		Tracks activities like sitting, lying down, eating, running, and crawling.
		Logs behaviors with timestamps for review and analysis.
	3.	Real-Time Performance:
		Uses advanced object detection and behavior classification algorithms.
		Processes live video streams from home-installed cameras.
	4.	Danger Zone Monitoring:
		Configures specific zones (e.g., staircases or kitchens) to monitor entry and exit events.
	5.	Behavior Reports:
		Maintains a timeline of activities for each tracked individual.
		Saves critical event snapshots for review.
	6.	Environmental Adaptation:
		Dynamically adjusts detection thresholds based on lighting and noise conditions.

Technologies Used

Datasets： Kinetics-400 / Custom Dataset

Models:   YOLOv5 / ResNet18


Frameworks: PyTorch / OpenCV / Shapely:


Other Tools: Twilio API / Email Libraries 

Project Configuration

Configuration File: config/config.yaml

The configuration file defines thresholds, paths, and danger zones:


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

FAQs

Q: How does the system detect dangerous events?

The system uses YOLOv5 to detect human targets and a ResNet18 classifier to recognize behaviors like falls. It dynamically adjusts detection thresholds for reliability.

Q: What happens when a fall is detected?

When a fall is detected:
	1.	An SMS or email is sent to the family member.
	2.	A snapshot of the event is saved for later review.

Q: Can I use a different camera?

Yes, the system supports both USB webcams and RTSP streams. Update the camera source in main.py. 
