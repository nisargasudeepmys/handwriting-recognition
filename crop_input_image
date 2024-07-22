import cv2
import os
from PIL import Image

def crop_alphabets_from_image(image_path, output_dir):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding the image to get a binary image
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Finding contours (bounding boxes)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Load the image using Pillow for cropping
    pil_image = Image.open(image_path)

    # Get bounding boxes and sort them by the x-coordinate (left edge)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    sorted_boxes = sorted(bounding_boxes, key=lambda box: box[0])

    # Loop through each sorted bounding box and crop the corresponding alphabet
    for i, (x, y, w, h) in enumerate(sorted_boxes):
        box = (x, y, x + w, y + h)

        # Crop the image using Pillow
        cropped_image = pil_image.crop(box)

        # Convert the image to RGB if it has an alpha channel
        if cropped_image.mode == 'RGBA':
            cropped_image = cropped_image.convert('RGB')

        # Save the cropped image
        cropped_image.save(f"{output_dir}/{os.path.basename(image_path).split('.')[0]}_alphabet_{i}.jpg")

# Define the input and output directories
input_dir = "/content/drive/My Drive/ML Project/input"  # Change this to your input directory
output_dir = "/content/drive/My Drive/ML Project/output"  # Change this to your output directory

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
        image_path = os.path.join(input_dir, filename)
        crop_alphabets_from_image(image_path, output_dir)

print("All images have been processed and cropped alphabets have been saved.")
