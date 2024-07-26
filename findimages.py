import cv2
import numpy as np

def find_and_circle_template(target_image_path, template_path, output_path):
    # Load the target image
    target_image = cv2.imread(target_image_path)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    target_edges = cv2.Canny(target_gray, 50, 200)

    # Load the template image
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    template_edges = cv2.Canny(template, 50, 200)
    (template_height, template_width) = template.shape[:2]

    # Perform template matching
    result = cv2.matchTemplate(target_edges, template_edges, cv2.TM_CCOEFF)
    (_, max_val, _, max_loc) = cv2.minMaxLoc(result)

    # Unpack the maximum location and compute the (x, y) coordinates of the bounding box
    (start_x, start_y) = max_loc
    (end_x, end_y) = (start_x + template_width, start_y + template_height)

    # Draw a bounding box around the detected result
    cv2.rectangle(target_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)

    # Save the result image
    cv2.imwrite(output_path, target_image)

# File paths
target_image_path = 'target_image.jpg'
template_path = 'gv_ui_battlefield_monster_mini_icon_0000.png'
output_path = 'output_image.jpg'

# Find and circle the template in the target image
find_and_circle_template(target_image_path, template_path, output_path)
