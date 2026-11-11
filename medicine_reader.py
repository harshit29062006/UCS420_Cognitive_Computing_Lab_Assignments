"""
UCS420 - Cognitive Computing
Assignment 9: Medicine Bottle Label Reader

Before running:
1. Install Tesseract OCR:
   macOS (Homebrew): brew install tesseract
2. Install Python packages:
   pip install -r requirements.txt
3. Put a clear medicine bottle photo named medicine.jpg
   in this same folder, or provide another image path when prompted.
"""

import os
import re
import shutil

import cv2
import pytesseract


def extract_dates(text):
    """Find common numeric date formats in the extracted label text."""
    pattern = r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
    return re.findall(pattern, text)


def main():
    # Find Tesseract automatically (works with common Homebrew locations).
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    elif os.path.exists("/opt/homebrew/bin/tesseract"):
        pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
    elif os.path.exists("/usr/local/bin/tesseract"):
        pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"
    else:
        print("Tesseract OCR was not found.")
        print("Install it with: brew install tesseract")
        return

    image_path = input("Enter image path (press Enter for medicine.jpg): ").strip()
    if not image_path:
        image_path = "medicine.jpg"

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not open '{image_path}'.")
        print("Place medicine.jpg in this folder or enter a valid image path.")
        return

    # Convert to grayscale and enlarge to help OCR read small text.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    processed = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # Extract printed text from the label.
    extracted_text = pytesseract.image_to_string(processed)

    print("\n========== MEDICINE LABEL READER ==========")
    print("\nExtracted text:")
    print(extracted_text.strip() if extracted_text.strip() else "No text detected.")

    dates = extract_dates(extracted_text)
    print("\nDetected dates:")
    if dates:
        for date in dates:
            print("-", date)
    else:
        print("No dates detected.")

    print("\nReminder: Check all extracted details against the original package.")
    print("This program does not verify medicine safety or recommend dosage.")

    # Show the original and processed images.
    cv2.imshow("Original Medicine Label", image)
    cv2.imshow("Processed Image for OCR", processed)
    print("\nPress any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
