# AI Person Disappear 👤✨

A real-time computer vision project that makes a person appear to disappear from a live camera feed.

The system detects and segments a person using YOLO11, generates a pixel-level mask, and replaces the person's region with a previously captured background reference.

> ⚠️ This is a computer vision demonstration project focused on real-time detection, segmentation, masking, and background reconstruction.

---

##  Project Overview

The idea behind this project is simple:

**What if a computer vision system could remove a person from a live camera feed while keeping the background looking natural?**

Instead of using a pre-existing background image, the system first captures the empty scene from the camera. When a person enters the frame, their region is detected and replaced using the captured background.

This creates the disappearing-person effect in real time.

---

##  How It Works

The complete pipeline is:

```text
Camera Input
     ↓
Person Detection
     ↓
Person Segmentation
     ↓
Mask Generation
     ↓
Mask Smoothing
     ↓
Background Reconstruction
     ↓
Final Output
Main Process
Camera Input
Captures the live video stream.
Person Detection & Segmentation
YOLO11 segmentation identifies people in the frame.
The model provides a pixel-level segmentation mask.
Background Capture
The system captures the scene before a person enters the frame.
This frame is used as the background reference.
Mask Processing
The person mask is processed and smoothed using OpenCV.
Background Reconstruction
The detected person region is replaced with the corresponding region from the captured background.
Real-Time Output
The processed frame is displayed as a live video feed.
🖥️ Desktop Version
The first version was developed as a desktop application using:
Python
OpenCV
YOLO11 Segmentation
The desktop application provides a real-time interface with controls for:
Capturing the background
Enabling/disabling disappearance mode
Viewing live camera output
Monitoring system status and FPS
🌐 Web Version
The same computer vision concept was also implemented as a browser-based application.
The web version uses:
Streamlit
WebRTC
OpenCV
YOLO11 Segmentation
Users can access the camera through the browser, capture the background, and test the disappearing-person effect.
🚀 Live Demo
Try the web version:
https://ai-person-disappear.streamlit.app/
🛠️ Tech Stack
Technology
Purpose
Python
Core development
OpenCV
Image & video processing
YOLO11 Segmentation
Person detection and segmentation
NumPy
Mask and image operations
Streamlit
Web interface
WebRTC
Browser-based camera streaming
📁 Project Structure
AI_Person_Disappaer/
│
├── app.py
├── requirements.txt
├── yolo11n-seg.pt
├── packages.txt
└── README.md
⚙️ Installation
1. Clone the repository
git clone https://github.com/shashiyaduc2-stack/AI_Person_Disappaer.git
cd AI_Person_Disappaer
2. Install dependencies
pip install -r requirements.txt
3. Run the web application
streamlit run app.py
The application will open in your browser.
🎮 How to Use
Start the application.
Allow camera access.
Start the camera.
Make sure the scene is empty.
Capture the background.
Enable disappearance mode.
Enter the camera frame.
Watch the system remove the detected person.
💡 What I Learned
This project helped me understand how different computer vision components work together in a real-time application.
Key concepts explored:
Object detection
Image segmentation
Pixel-level masking
Real-time video processing
Background reconstruction
OpenCV image processing
WebRTC camera streaming
Streamlit application development
Deploying a computer vision application for web access
🚧 Limitations
The system works best when:
The camera remains relatively stationary.
The background does not change significantly.
The captured background is similar to the live scene.
The person remains within the camera's view.
Because the background behind a person is not actually visible while they are standing there, the system uses the previously captured background reference rather than recovering the original hidden pixels.
🔮 Future Improvements
Possible improvements include:
Better background reconstruction
More accurate edge handling
Improved performance on low-end hardware
Support for multiple people
Better handling of moving backgrounds
More advanced real-time inpainting techniques
Improved web deployment and scalability
📸 Demo
Desktop Version
The desktop version performs the disappearing effect directly from the live camera feed.
Web Version
The browser-based version provides the same core computer vision functionality through a web interface.
👩‍💻 Author
Shashi
B.Tech CSE — Artificial Intelligence & Machine Learning
Interested in:
Artificial Intelligence
Machine Learning
Computer Vision
Data Science
⭐ If you find this project interesting
Feel free to explore the repository, try the live demo, and share your feedback!
