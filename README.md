# Air Canvas

This is an Air Canvas project implemented in Python using OpenCV, Numpy, and Google's Mediapipe. It tracks your hands to allow you to draw in the air.

> **Note:** This project is created primarily for **learning purposes** and exploring computer vision techniques using Python and Mediapipe. It is a fun experimental project and not intended for production pipelines.

## Features
- **Two-Hand Support:** Draw with both your left and right hands collaboratively!
- **Color Selection:** Choose between Blue, Green, Red, and Yellow by pointing at the menu.
- **Adjustable Brush Size:** Use the `SIZE +` and `SIZE -` buttons to dynamically change line thickness.
- **Smart Gestures:** 
  - **Draw:** Extend *only* your index finger to write or draw.
  - **Erase:** Clench your hand into a **fist** to turn it into an eraser that wipes away drawn lines cleanly.
  - **Hover:** Open your hand or extend multiple fingers to move around safely without drawing.
- **Clear Canvas:** Point to the `CLEAR` button to wipe all drawings instantly.

## Requirements
Make sure you have a webcam connected to your computer.
Dependencies are listed in `requirements.txt`.

## How to Run
1. Install the dependencies (if you haven't already):
    ```bash
    pip install -r requirements.txt
    ```
2. Run the main script:
    ```bash
    py main.py
    ```

## Usage
- **Draw:** Raise your hand and extend **only your index finger**. The camera detects your fingertip and plots your strokes.
- **Erase:** Make a **fist** and glide it over your strokes to erase them like a whiteboard eraser.
- **Select Tools:** Move your index finger into the UI boxes at the top of the screen to switch colors or adjust the brush size.
- **Quit:** Press `q` while the application window is focused to exit.
