import cv2
import mediapipe as mp
import pyautogui
import time

# Enable failsafe for PyAutoGUI
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

try:
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not access camera")
    
    # Initialize hand detection
    hand_detector = mp.solutions.hands.Hands()
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()
    index_y = 0

    while True:
        # Read and process frame
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape
        
        # Detect hands
        results = hand_detector.process(rgb_frame)
        hands = results.multi_hand_landmarks
        
        if hands:
            for hand in hands:
                drawing_utils.draw_landmarks(frame, hand)
                landmarks = hand.landmark
                
                for id, landmark in enumerate(landmarks):
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)
                    
                    # Index finger (for mouse movement)
                    if id == 8:
                        cv2.circle(img=frame, center=(x, y), radius=10, color=(255, 0, 0), thickness=-1)
                        index_x = screen_width / frame_width * x
                        index_y = screen_height / frame_height * y
                        pyautogui.moveTo(index_x, index_y)
                    
                    # Thumb (for clicking)
                    if id == 4:
                        cv2.circle(img=frame, center=(x, y), radius=10, color=(255, 0, 0), thickness=-1)
                        thumb_x = screen_width / frame_width * x
                        thumb_y = screen_height / frame_height * y
                        # Click when thumb and index finger are close
                        if abs(thumb_y - index_y) < 20:
                            pyautogui.click()
                            time.sleep(1)
        
        # Display output
        cv2.imshow('Virtual Mouse', frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    # Clean up resources
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
