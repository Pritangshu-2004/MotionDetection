import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Read the first frame
ret, frame1 = cap.read()

# Convert to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

# Create VideoWriter to save the processed video to file
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('motion_output.avi', fourcc, 20.0, (640, 480))

frame_count = 0  # To track the number of frames for saving individual frames

while True:
    # Read a new frame
    ret, frame2 = cap.read()

    # Convert to grayscale
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    # Compute the absolute difference between the current frame and the previous one
    frame_delta = cv2.absdiff(gray1, gray2)

    # Threshold the delta to get the regions with motion
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Flag to check if motion was detected
    motion_detected = False

    # Draw rectangles around moving objects
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Ignore small movements
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True

    # Save frame with motion detection to a file
    if motion_detected:
        frame_count += 1
        cv2.imwrite(f"motion_frame_{frame_count}.png", frame2)

    # Display the resulting frame
    cv2.imshow("Motion Detection", frame2)

    # Write the frame to the output video
    out.write(frame2)

    # Update the previous frame
    gray1 = gray2.copy()

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
out.release()  # Close the video file
cv2.destroyAllWindows()
