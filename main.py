import cv2
import numpy as np

def main():
    # Initialize the webcam feed (0 is the built-in default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Computer Vision Application...")
    print("Press 'q' on your keyboard to exit the application.")

    # Read the first frame for background reference (Motion Detection / Segmentation)
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Failed to grab initial frame.")
        return
    
    # 1. Image Operations: Resize and convert to grayscale
    frame1 = cv2.resize(frame1, (640, 480))
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Resize current frame for smooth processing
        frame2 = cv2.resize(frame2, (640, 480))
        display_frame = frame2.copy()

        # ==========================================
        # TECHNIQUE 1: Image Operations (Grayscale & Blur)
        # ==========================================
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

        # ==========================================
        # TECHNIQUE 2: Image Segmentation (Frame Differencing & Thresholding)
        # ==========================================
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours of moving objects
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < 1200:  # Ignore tiny noise/flicker
                continue
            
            motion_detected = True
            # Draw bounding box for motion
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(display_frame, "Motion Object", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ==========================================
        # TECHNIQUE 3: Image Analysis & Transformation (Canny Edge Detection)
        # ==========================================
        edges = cv2.Canny(gray2, 100, 200)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Place a small live edge-detection preview window on the main display frame
        h_preview, w_preview = 120, 160
        edges_resized = cv2.resize(edges_colored, (w_preview, h_preview))
        display_frame[10:10+h_preview, display_frame.shape[1]-w_preview-10:display_frame.shape[1]-10] = edges_resized
        
        # Label the preview box
        cv2.putText(display_frame, "Canny Edges", (display_frame.shape[1] - w_preview - 10, h_preview + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Display general status text on screen
        status_text = "Status: Motion Detected!" if motion_detected else "Status: Monitoring..."
        status_color = (0, 0, 255) if motion_detected else (0, 255, 0)
        cv2.putText(display_frame, status_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Show the final processed window
        cv2.imshow('CSC 126 Computer Vision Project', display_frame)

        # Update reference frame for continuous motion tracking
        gray1 = gray2.copy()

        # Press 'q' to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()