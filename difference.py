import cv2
import numpy as np

def compare_images(image1_path, image2_path, output_path):
    # Load the two images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference between the two images
    diff = cv2.absdiff(gray1, gray2)
    
    # Threshold the difference image
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Find contours of the differences
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the contours on the original image
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Ignore small contours
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Save the result image
    cv2.imwrite(output_path, image1)

# Example usage
compare_images('screenshot2.png', 'screenshot3.png', 'output.png')