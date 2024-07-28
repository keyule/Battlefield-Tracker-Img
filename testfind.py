import cv2
import numpy as np

def check_exists_multiple(large_image_path, template_path, top_n, min_distance=10, threshold=0.7, method=cv2.TM_CCOEFF_NORMED):
    large_image = cv2.imread(large_image_path)
    template = cv2.imread(template_path)
    template_height, template_width, _ = template.shape
    result = cv2.matchTemplate(large_image, template, method)
    sorted_results = []

    while len(sorted_results) < top_n:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Check if the similarity score is above the threshold
        if max_val < threshold:
            break
        
        # Check if the new location is sufficiently far from existing locations
        is_far_enough = True
        for _, existing_loc in sorted_results:
            distance = np.sqrt((existing_loc[0] - max_loc[0]) ** 2 + (existing_loc[1] - max_loc[1]) ** 2)
            if distance < min_distance:
                is_far_enough = False
                break
        
        if is_far_enough:
            sorted_results.append((max_val, max_loc))
        
        # Set the found location to a low value to find the next highest match
        result[max_loc[1], max_loc[0]] = -1

        # Break if there are no more valid matches
        if max_val < 0:
            break

    # Calculate the middle coordinates for the top N found locations
    middle_coords = []
    for score, pt in sorted_results:
        middle_x = pt[0] + template_width / 2
        middle_y = pt[1] + template_height / 2
        middle_coords.append((middle_x, middle_y, pt[0], pt[1], template_width, template_height))

    return middle_coords

def find_and_circle_template(target_image_path, template_path, output_path, top_n=5, min_distance=10, threshold=0.7):
    methods = {
        'TM_CCOEFF': cv2.TM_CCOEFF, 
        'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED, 
        'TM_CCORR': cv2.TM_CCORR,
        'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED, 
        'TM_SQDIFF': cv2.TM_SQDIFF, 
        'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED
    }

    for method_name, method in methods.items():
        target_image = cv2.imread(target_image_path)
        coordinates = check_exists_multiple(target_image_path, template_path, top_n, min_distance, threshold, method)

        for coord in coordinates:
            middle_x, middle_y, start_x, start_y, template_width, template_height = coord
            end_x, end_y = start_x + template_width, start_y + template_height
            cv2.rectangle(target_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            cv2.circle(target_image, (int(middle_x), int(middle_y)), 10, (0, 255, 0), 2)

        method_output_path = output_path.replace('.jpg', f'_{method_name}.jpg')
        cv2.imwrite(method_output_path, target_image)

# File paths
target_image_path = 'rift_1.png'
template_path = 'wolf_small.png'
output_path = 'output_image.jpg'

# Find and circle the top N matches using different methods
find_and_circle_template(target_image_path, template_path, output_path, top_n=5, min_distance=10, threshold=0.7)
