import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def edge_detect_image(image):
    edges = cv2.Canny(image, 50, 200)
    return edges

def check_exists_multiple(large_image_path, template_path, top_n, min_distance=10, threshold=0.7):
    large_image = preprocess_image(large_image_path)
    template = preprocess_image(template_path)
    template_height, template_width = template.shape

    large_edges = edge_detect_image(large_image)
    template_edges = edge_detect_image(template)

    result = cv2.matchTemplate(large_edges, template_edges, cv2.TM_CCOEFF)
    sorted_results = []

    while len(sorted_results) < top_n:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val < threshold:
            break

        is_far_enough = True
        for _, existing_loc in sorted_results:
            distance = np.sqrt((existing_loc[0] - max_loc[0]) ** 2 + (existing_loc[1] - max_loc[1]) ** 2)
            if distance < min_distance:
                is_far_enough = False
                break

        if is_far_enough:
            sorted_results.append((max_val, max_loc))

        result[max_loc[1], max_loc[0]] = -1

        if max_val < 0:
            break

    middle_coords = []
    for score, pt in sorted_results:
        middle_x = pt[0] + template_width / 2
        middle_y = pt[1] + template_height / 2
        middle_coords.append((middle_x, middle_y, pt[0], pt[1], template_width, template_height))

    return middle_coords

def find_and_circle_template(target_image_path, template_path, output_path, top_n=5, min_distance=10, threshold=0.7):
    target_image = cv2.imread(target_image_path)
    coordinates = check_exists_multiple(target_image_path, template_path, top_n, min_distance, threshold)

    for coord in coordinates:
        middle_x, middle_y, start_x, start_y, template_width, template_height = coord
        end_x, end_y = start_x + template_width, start_y + template_height
        cv2.rectangle(target_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
        cv2.circle(target_image, (int(middle_x), int(middle_y)), 10, (0, 255, 0), 2)

    cv2.imwrite(output_path, target_image)

# File paths
target_image_path = 'screenshot.png'
template_path = 'gv_ui_battlefield_monster_mini_icon_0000.png'
output_path = 'output_image.jpg'

# Find and circle the top N matches
find_and_circle_template(target_image_path, template_path, output_path, top_n=5, min_distance=10, threshold=0.7)
