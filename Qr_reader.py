import cv2
from pyzbar.pyzbar import decode
import time

def read_qr_from_frame(frame):
    """
    Scans the given frame for a QR code and returns its plain text contents.
    """
    qrcodes = decode(frame)
    # Iterate through detected QR codes
    for code in qrcodes:
        try:
            # Since we assume plain text, simply decode using utf-8.
            data = code.data.decode('utf-8')
            name,purpose_of_visit, time = filter_qr_data(data)
            return name,purpose_of_visit, time
        except Exception as e:
            print("Error decoding QR code:", e)
    return None, None,None

def filter_qr_data(qr_data):
    """this function is used to filter the qr data and extract purpose of visit and time"""
    qr_data = qr_data.split(",")
    if len(qr_data) == 3:
        name,purpose_of_visit, time = qr_data
        return name,purpose_of_visit, time
    return None, None,None

def start_qr_reader():
    """
    Opens the webcam and continuously scans for a QR code.
    Returns the QR code's plain text string when found.
    """
    cap = cv2.VideoCapture(0)
    print("Starting QR code scanner. Please show your QR code to the camera...")
    qr_text = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        
        # Try reading a QR from the frame.
        qr_text = read_qr_from_frame(frame)
        
        # If a QR code is detected, overlay a message and break out after a short delay.
        if qr_text[0] is not None:
            cv2.putText(frame, "QR Code Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("QR Scanner", frame)
            cv2.waitKey(1000)  # Show the message for 1 second.
            break

        cv2.imshow("QR Scanner", frame)
        # Press 'q' to exit manually.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return qr_text

if __name__ == "__main__":
    qr_data = start_qr_reader()
    if qr_data:
        print("QR Code data:", qr_data)
    else:
        print("No QR code detected.")
