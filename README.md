# Face-recognition-and-tracking-using-CCTV-in-a-theft
Combine age_net.caffemodel parts as one and name it as age_net.caffemodel


**Project Description: CCTV Camera with Intelligent Theft Detection and Alert System**

In this project, I developed an advanced CCTV camera module equipped with intelligent theft detection and alert capabilities. The system is designed to enhance security by capturing and tracking human faces during a theft incident, subsequently notifying the residents or any emergency contacts set in the directory.

### Key Features:

1. **Face Detection and Tracking:**
   - Utilized pre-trained machine learning models for robust face detection and recognition.
   - Implemented a face tracking module to continuously monitor and follow detected faces within the camera's field of view.

2. **Theft Detection:**
   - Integrated motion detection algorithms to identify suspicious activities indicating potential theft.
   - Employed logic to differentiate between normal movement and theft-related behavior.

3. **Alert System:**
   - Configured the system to send real-time alerts to residents or designated emergency contacts.
   - Alerts include captured images of the intruder's face and relevant details of the incident.

### Technologies and Tools Used:

- **Python:** Main programming language for implementing face detection, tracking algorithms, and motion detection.
- **OpenCV:** Library for computer vision tasks such as face detection, image processing, and tracking.
- **Arduino:** Microcontroller used to manage hardware components such as cameras, sensors, and communication modules.
- **Pre-trained Models:** Leveraged models like Haar cascades and deep learning-based models for accurate face detection.

### Implementation Details:

1. **Face Detection and Tracking Module:**
   - Loaded pre-trained face detection models using OpenCV.
   - Developed a tracking algorithm to follow detected faces within the camera's view.

2. **Motion Detection:**
   - Implemented motion detection using background subtraction techniques to identify unusual movements.

3. **Alert Mechanism:**
   - Integrated communication modules (e.g., GSM module) with Arduino to send alerts.
   - Developed a Python script to capture images and send notifications via email or SMS.

### Code Components:

- **Python Code:**
  - Face detection and tracking script.
  - Motion detection logic.
  - Image capturing and alert sending functions.

- **Arduino Code:**
  - Interfacing with cameras and sensors.
  - Handling communication with the notification system.

### Outcome:

The project resulted in a functional CCTV camera system capable of detecting and tracking intruders, capturing their images, and promptly alerting designated contacts. This enhanced security setup provides an effective deterrent against theft and ensures quick response during incidents.
