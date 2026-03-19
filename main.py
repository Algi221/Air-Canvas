import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import os

# Download the model if not present
if not os.path.exists('hand_landmarker.task'):
    import urllib.request
    print("Downloading hand_landmarker.task...")
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(url, 'hand_landmarker.task')

# Initialize Mediapipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

landmarker = HandLandmarker.create_from_options(options)

# Setup video capture
cap = cv2.VideoCapture(0)

# Colors in BGR format
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)] # Blue, Green, Red, Yellow

class HandState:
    def __init__(self):
        self.strokes = []
        self.colorIndex = 0
        self.brushSize = 5
        self.cooldown = 0

hands_state = {
    'Left': HandState(),
    'Right': HandState()
}

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1) # Flip horizontally for a mirror effect
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Draw UI on the frame
    frame = cv2.rectangle(frame, (40,1), (140,65), (122,122,122), -1)
    frame = cv2.rectangle(frame, (160,1), (255,65), (255,0,0), -1)
    frame = cv2.rectangle(frame, (275,1), (370,65), (0,255,0), -1)
    frame = cv2.rectangle(frame, (390,1), (485,65), (0,0,255), -1)
    frame = cv2.rectangle(frame, (505,1), (600,65), (0,255,255), -1)
    
    frame = cv2.rectangle(frame, (40,75), (140,120), (122,122,122), -1)
    frame = cv2.rectangle(frame, (160,75), (255,120), (122,122,122), -1)
    
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)
    
    cv2.putText(frame, "SIZE +", (65, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "SIZE -", (185, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


    # Hand Tracking with new Tasks API
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=framergb)
    result = landmarker.detect(mp_image)
    
    if result and len(result.hand_landmarks) > 0:
        h, w, c = frame.shape
        detected_hands = []
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            hand_category = "Left"
            if result.handedness and len(result.handedness) > i:
                hand_category = result.handedness[i][0].category_name
            
            detected_hands.append(hand_category)
            state = hands_state[hand_category]
            
            # Calculate screen coordinates
            landmarks = []
            for lm in hand_landmarks:
                lmx = int(lm.x * w)
                lmy = int(lm.y * h)
                landmarks.append([lmx, lmy])
                cv2.circle(frame, (lmx, lmy), 2, (0, 0, 255), -1)
                
            fore_finger = (landmarks[8][0], landmarks[8][1])
            center = fore_finger
            palm_center = (landmarks[9][0], landmarks[9][1])
            
            cv2.circle(frame, center, 4, (0,255,0), -1)
            
            # Check for open palm (mopping / erasing)
            dist_index = np.hypot(landmarks[8][0] - landmarks[0][0], landmarks[8][1] - landmarks[0][1])
            dist_middle = np.hypot(landmarks[12][0] - landmarks[0][0], landmarks[12][1] - landmarks[0][1])
            dist_ring = np.hypot(landmarks[16][0] - landmarks[0][0], landmarks[16][1] - landmarks[0][1])
            dist_pinky = np.hypot(landmarks[20][0] - landmarks[0][0], landmarks[20][1] - landmarks[0][1])
            
            d_mcp_idx = np.hypot(landmarks[5][0] - landmarks[0][0], landmarks[5][1] - landmarks[0][1])
            d_mcp_mid = np.hypot(landmarks[9][0] - landmarks[0][0], landmarks[9][1] - landmarks[0][1])
            d_mcp_rng = np.hypot(landmarks[13][0] - landmarks[0][0], landmarks[13][1] - landmarks[0][1])
            d_mcp_pnk = np.hypot(landmarks[17][0] - landmarks[0][0], landmarks[17][1] - landmarks[0][1])
            
            is_index_up = dist_index > 1.3 * d_mcp_idx
            is_middle_up = dist_middle > 1.3 * d_mcp_mid
            is_ring_up = dist_ring > 1.3 * d_mcp_rng
            is_pinky_up = dist_pinky > 1.3 * d_mcp_pnk
            
            fingers_open = sum([is_index_up, is_middle_up, is_ring_up, is_pinky_up])
            
            if state.cooldown > 0:
                state.cooldown -= 1
                
            if fingers_open == 0:
                # Ngepal (Fist) -> Eraser mode
                if palm_center[1] > 125: # Leave space for UI interactions
                    cv2.circle(frame, palm_center, 40, (122, 122, 122), 2) # feedback UI
                    
                    if len(state.strokes) == 0 or state.strokes[-1]['color'] != (255,255,255):
                        state.strokes.append({'color': (255,255,255), 'thickness': 60, 'points': deque(maxlen=512)})
                    state.strokes[-1]['points'].appendleft(palm_center)
                continue
            elif is_index_up and fingers_open == 1:
                # Index finger is up -> check UI and draw
                if center[1] <= 65:
                    if 40 <= center[0] <= 140: # Clear UI
                        for h_name in ['Left', 'Right']:
                            hands_state[h_name].strokes = []
                    elif 160 <= center[0] <= 255: state.colorIndex = 0
                    elif 275 <= center[0] <= 370: state.colorIndex = 1
                    elif 390 <= center[0] <= 485: state.colorIndex = 2
                    elif 505 <= center[0] <= 600: state.colorIndex = 3
                    
                    if len(state.strokes) > 0 and len(state.strokes[-1]['points']) > 0:
                        state.strokes.append({'color': colors[state.colorIndex], 'thickness': state.brushSize, 'points': deque(maxlen=512)})
                elif 75 <= center[1] <= 120:
                    # Size UI interaction
                    if state.cooldown == 0:
                        if 40 <= center[0] <= 140:
                            state.brushSize = min(50, state.brushSize + 1)
                            state.cooldown = 2 # Add cooldown to prevent too fast increment
                        elif 160 <= center[0] <= 255:
                            state.brushSize = max(2, state.brushSize - 1)
                            state.cooldown = 2
                else:
                    # Drawing
                    if len(state.strokes) == 0 or state.strokes[-1]['color'] != colors[state.colorIndex] or state.strokes[-1]['thickness'] != state.brushSize:
                        state.strokes.append({'color': colors[state.colorIndex], 'thickness': state.brushSize, 'points': deque(maxlen=512)})
                    state.strokes[-1]['points'].appendleft(center)
            else:
                # Other gestures -> break line
                if len(state.strokes) > 0 and len(state.strokes[-1]['points']) > 0:
                    state.strokes.append({'color': colors[state.colorIndex], 'thickness': state.brushSize, 'points': deque(maxlen=512)})
                continue
            
        # for hands not detected, break the line
        for hand_category in ['Left', 'Right']:
            if hand_category not in detected_hands:
                state = hands_state[hand_category]
                if len(state.strokes) > 0 and len(state.strokes[-1]['points']) > 0:
                    state.strokes.append({'color': colors[state.colorIndex], 'thickness': state.brushSize, 'points': deque(maxlen=512)})
    else:
        # Hand was removed from screen, break the line
        for hand_category in ['Left', 'Right']:
            state = hands_state[hand_category]
            if len(state.strokes) > 0 and len(state.strokes[-1]['points']) > 0:
                state.strokes.append({'color': colors[state.colorIndex], 'thickness': state.brushSize, 'points': deque(maxlen=512)})

    # Print current active sizes for visual feedback from either hand
    cv2.putText(frame, f"L Size: {hands_state['Left'].brushSize}", (275, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"R Size: {hands_state['Right'].brushSize}", (390, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Prepare a blank canvas for drawing strokes
    canvasForLines = np.zeros((480, 640, 3), dtype=np.uint8) + 255
    
    # Draw the lines on canvas
    for hand_category in ['Left', 'Right']:
        state = hands_state[hand_category]
        for stroke in state.strokes:
            pts = stroke['points']
            col = stroke['color']
            thk = stroke['thickness']
            for k in range(1, len(pts)):
                if pts[k-1] is None or pts[k] is None:
                    continue
                cv2.line(canvasForLines, pts[k-1], pts[k], col, thk)
                
    # Create mask to overlay lines over the camera frame transparently
    mask_gray = cv2.cvtColor(canvasForLines, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask_gray, 254, 255, cv2.THRESH_BINARY_INV)
    
    colored_lines = cv2.bitwise_and(canvasForLines, canvasForLines, mask=mask)
    inv_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(frame, frame, mask=inv_mask)
    
    frame = cv2.add(background, colored_lines)
    
    # Recreate paintWindow from scratch to synchronize UI dynamically
    paintWindow = np.zeros((480, 640, 3), dtype=np.uint8) + 255
    paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
    paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), (255,0,0), 2)
    paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), (0,255,0), 2)
    paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), (0,0,255), 2)
    paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), (0,255,255), 2)
    # Size UI in paintWindow
    paintWindow = cv2.rectangle(paintWindow, (40, 75), (140, 120), (0,0,0), 2)
    paintWindow = cv2.rectangle(paintWindow, (160, 75), (255, 120), (0,0,0), 2)
    
    cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "SIZE +", (65, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "SIZE -", (185, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    
    cv2.putText(paintWindow, f"L Size: {hands_state['Left'].brushSize}", (275, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, f"R Size: {hands_state['Right'].brushSize}", (390, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Overlay colored lines from canvas on paintWindow layout
    paintWindow_bg = cv2.bitwise_and(paintWindow, paintWindow, mask=inv_mask)
    paintWindow = cv2.add(paintWindow_bg, colored_lines)
                
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)
    
    # Check for 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
