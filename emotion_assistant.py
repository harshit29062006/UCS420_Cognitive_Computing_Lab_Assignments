"""
UCS420 - Cognitive Computing
Assignment 9: Emotion-Aware Cognitive Assistant

Install packages:
    pip install -r requirements.txt

Run:
    python emotion_assistant.py

Allow camera access if macOS asks. Press Q to quit.
DeepFace uses a pretrained model; its first run may download model files.
"""

import cv2
from deepface import DeepFace


def get_response(emotion):
    """Return a simple supportive response for the detected expression."""
    responses = {
        "happy": "You look happy! Keep smiling.",
        "sad": "You may be having a difficult moment. Would a short break help?",
        "angry": "Consider taking a slow, calming breath.",
        "fear": "Try to pause and take a few comfortable breaths.",
        "surprise": "You look surprised!",
        "disgust": "Take a moment if you need one.",
        "neutral": "You look calm and relaxed."
    }

    return responses.get(emotion, "Thank you for using the assistant.")


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not access the webcam.")
        print("Check camera permissions and close other apps using the camera.")
        return

    print("Emotion-Aware Assistant started.")
    print("Press Q in the camera window to exit.")
    print("Note: facial expressions are not a reliable measure of a person's true feelings.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read a frame from the webcam.")
            break

        frame = cv2.resize(frame, (640, 480))

        try:
            result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True
            )

            # DeepFace normally returns a list of face-analysis results.
            if isinstance(result, list):
                result = result[0]

            emotion_scores = result.get("emotion", {})
            emotion = result.get("dominant_emotion", "unknown")
            confidence = emotion_scores.get(emotion, 0.0)

            cv2.putText(
                frame,
                f"Expression: {emotion.upper()}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Model score: {confidence:.1f}%",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            response = get_response(emotion)
            cv2.putText(
                frame,
                response[:65],
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1
            )

        except Exception as error:
            cv2.putText(
                frame,
                "Face analysis unavailable",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            print("Analysis message:", error)

        cv2.imshow("Emotion-Aware Cognitive Assistant", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
