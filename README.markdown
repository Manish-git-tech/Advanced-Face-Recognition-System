# 🚀 Face Recognition System with Adaptive Learning

## 🔍 Overview
This Face Recognition System is designed to identify and log employees in a secure environment. The system utilizes deep learning techniques for face detection and recognition, allowing for seamless entry and exit logging. It includes features for employee registration, log management, and basic anti-spoofing measures.

## 🌟 Features
- **👤 Employee Registration**: Register new employees by capturing multiple face images.
- **📸 Real-time Face Recognition**: Detect and recognize multiple faces in real-time using a webcam.
- **📜 Entry/Exit Logging**: Automatically log employee entry and exit times based on face recognition.
- **📝 Manual Log Entry**: Administrators can manually add or delete logs by providing the employee’s institute ID and the entry/exit time.
- **👥 Employee Management**:
  - View, delete, and manage employee records.
  - Search for an employee by their institute ID.
  - Delete employees by selecting them and pressing the delete button.
  - Add new employees by uploading their photos.
- **📑 Log Management**:
  - Show all previous logs with a search option to filter logs by date.
  - Admin can select logs to delete and press the delete button.
- **📌 Profile Management**:
  - Show employee profile details upon clicking their entry in the employee list.
  - Display institute ID, name, number of entries, number of exits, last log entry (type and timestamp), and current status (inside or outside campus).
- **📈 Adaptive Learning**: Continuously update face embeddings to improve recognition performance over time.

## 🛠️ Admin Panel
![alt text](<data/Project Screenshot/Admin_panel_1.png>)
The admin panel pro!vides a streamlined interface for managing employees and log records efficiently:
- **👀 View Employees**: Displays a list of employees with their ID, institute ID, and name. Clicking on an employee opens a new window showing detailed information, including their profile photo, last log details, and campus status.
- **➕ Add Employee**: Upload photos to register new employees into the system.
- **📂 Manage Logs**:
  - View all log records with search functionality.
  - Manually enter new log records using an institute ID and time.
  - Delete log entries by selecting specific logs and pressing the delete button.
- **🗑️ Delete Employees**: Similar to log deletion, administrators can search for an employee by institute ID, select them, and delete them.

    ![alt text](<data/Project Screenshot/Admin_panel_3.png>)
## 📦 Requirements
- Python 3.x
- Required libraries:
  - OpenCV
  - NumPy
  - Pillow
  - Dlib
  - Streamlit
  - InsightFace

## 🔧 Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/face_recognition_system.git
   cd face_recognition_system
   ```

2. Create a new conda environment:
   ```
   conda create -n face_recognition_env python=3.8
   conda activate face_recognition_env
   ```

3. Install required packages:
   ```
   pip install opencv-python numpy pillow dlib imutils streamlit insightface
   ```

4. Download the Dlib shape predictor model:
   ```
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
   ```

5. Place the `shape_predictor_68_face_landmarks.dat` file in the project directory.

## 🚀 Usage
1. Start the admin panel:
   ```
   streamlit run admin_app.py
   ```

2. Use the GUI to manage employee records, view logs, and register new employees.

3. To run the face recognition system:
   ```
   python recognition_app.py
   ```

## 📂 Directory Structure
```
face_recognition_system/
│
├── config.py                  # Configuration settings for the application.
├── database_handler.py         # Database management functions.
├── employee_registrar.py       # Employee registration logic.
├── face_processor.py           # Face detection and recognition logic.
├── recognition_app.py          # Main application for real-time face recognition.
├── admin_app.py                # Streamlit app for administration tasks.
└── logs/                        # Stores log files.
```

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgments
- 🧠 **InsightFace** for pre-trained models.
- 🏷️ **Dlib** for facial landmark detection.
- 🎨 **Streamlit** for creating a user-friendly interface.

## 📧 Contact Information
For any inquiries or issues, please contact [your.email@example.com].

