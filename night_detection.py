import cv2
import os
import pytesseract
from PIL import Image

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained license plate cascade classifier
plate_cascade = cv2.CascadeClassifier(os.path.join(current_dir, '/home/mamatha/Desktop/license_plate_detection_night/india_license_plate.xml'))

# Function to detect and highlight license plates in a frame of a video
def detect_and_highlight_plate(frame, save_path, image_counter, saved_plates):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plate_rects = plate_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7)

    for i, (x, y, w, h) in enumerate(plate_rects):
        plate_id = (x, y, w, h)
        if plate_id not in saved_plates:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)
            plate_region = frame[y:y+h, x:x+w]
            filename = os.path.join(save_path, f"plate_{image_counter}_{i}.png")
            cv2.imwrite(filename, plate_region)
            print(f"Plate image saved: {filename}")
            saved_plates.add(plate_id)

    return frame

# Function to process a video
def process_video(video_path):
    video_detect = cv2.VideoCapture(video_path)
    output_directory = os.path.join(current_dir, 'detected_frames')
    os.makedirs(output_directory, exist_ok=True)
    image_counter = 1
    saved_plates = set()

    while True:
        ret, frame = video_detect.read()
        if not ret:
            break

        frame = cv2.resize(frame, (700, 600))
        frame = detect_and_highlight_plate(frame, output_directory, image_counter, saved_plates)
        cv2.imshow('License Plate Detection-Night', frame)
        key = cv2.waitKey(1) & 0xFF
        if key in {ord('q'), 27}:
            break
        image_counter += 1

    video_detect.release()
    cv2.destroyAllWindows()

# Menu to choose between image and video processing...

    # Directory where the output images are saved
    directory = os.path.join(current_dir, 'detected_frames')
    files = os.listdir(directory)

    # List of image files in the output directory
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Setting the Tesseract executable path based on the operating system
    if os.name == 'posix':  # For Linux or Mac OS
        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    else:  # For Windows
        pytesseract.pytesseract.tesseract_cmd = os.path.join(current_dir, 'tesseract.exe')

    # Specify the output text file for storing OCR results
    output_file = os.path.join(current_dir, 'number_plates.txt')

    # Open the output file in write mode
    with open(output_file, 'w') as output:
        # Iterate through each image file and perform OCR
        for image_file in image_files:
            image_path = os.path.join(directory, image_file)
            try:
                # Open the image using the PIL library
                image = Image.open(image_path)
                # Perform OCR on the image to extract text
                text = pytesseract.image_to_string(image)
                # Write the results to the output file
                output.write(f"'{image_file}' : '{text.strip()}'\n")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

    # Function to remove lines from a file based on length
    def remove_lines_by_length(file_path):
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
            first_line_length = len(lines[0].strip())
            updated_lines = [line for line in lines if len(line.strip()) != first_line_length]

            # Specify the path for the updated output file
            output_file_path = os.path.join(current_dir, 'number_plates.txt')
            
            # Write the updated lines to the new output file
            with open(output_file_path, 'w') as output_file:
                output_file.writelines(updated_lines)
            print(f"Lines with the same length removed. Updated content saved to {output_file_path}")
        except Exception as e:
            print(f"Error removing lines by length: {e}")

    # Specify the path to the original output
    file_path = os.path.join(current_dir, 'number_plates.txt')
    # Call the function to remove lines with the same length
    remove_lines_by_length(file_path)

    # Specify the path to the final result file
    result = os.path.join(current_dir, 'number_plates.txt')

    # Try to open and print the content of the result file
    try:
        with open(result, 'r') as file:
            for line in file:
                print(line.strip())
    except FileNotFoundError:
        print(f"File not found: {result}")
    except Exception as e:
        print(f"Error reading the file: {e}")

# Function to detect and highlight license plates in an image
def detect_and_highlight_plate_image(image_path, save_path, image_counter, saved_plates):
    # Read the image using OpenCV
    frame = cv2.imread(image_path)
    
    # Convert the image to grayscale for better processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect license plate regions in the image
    plate_rects = plate_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7)

    # Iterate through the detected plates and highlight/save them
    for i, (x, y, w, h) in enumerate(plate_rects):
        plate_id = (x, y, w, h)
        if plate_id not in saved_plates:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)  # Draw a rectangle around the plate
            plate_region = frame[y:y+h, x:x+w]  # Extract the region of interest (ROI) containing the plate
            filename = os.path.join(save_path, f"plate_{image_counter}_{i}.png")
            cv2.imwrite(filename, plate_region)  # Save the plate region as an image
            print(f"Plate image saved: {filename}")
            saved_plates.add(plate_id)

    return frame
# Rest of the code remains unchanged...

# Function to process an image
def process_image(image_path):
    # Specify the directory to save the output images
    output_directory = os.path.join(current_dir, 'license_plate_detection_night')
    os.makedirs(output_directory, exist_ok=True)

    # Initialize a set to keep track of saved plates to avoid duplicates
    saved_plates = set()
    # Counter to keep track of the processed images
    image_counter = 1

    # Call the function to detect and highlight license plates in the image
    detect_and_highlight_plate_image(image_path, output_directory, image_counter, saved_plates)

    # Directory where the output images are saved
    directory = os.path.join(current_dir, 'license_plate_detection_night')
    # List all files in the directory with specific extensions
    files = os.listdir(directory)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Specify the path to the Tesseract OCR executable
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    # Specify the output text file for storing OCR results
    output_file = os.path.join(current_dir, 'image_output.txt')

    # Open the output file in write mode
    with open(output_file, 'w') as output:
        # Iterate through each image file and perform OCR
        for image_file in image_files:
            image_path = os.path.join(directory, image_file)
            try:
                # Open the image using the PIL library
                image = Image.open(image_path)
                # Perform OCR on the image to extract text
                text = pytesseract.image_to_string(image)
                # Write the results to the output file
                output.write(f"'{image_file}' : '{text.strip()}'\n")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

    # Specify the path to the result file
    result = os.path.join(current_dir, 'image_output.txt')

    # Try to open and print the content of the result file
    try:
        with open(result, 'r') as file:
            for line in file:
                print(line.strip())
    except FileNotFoundError:
        print(f"File not found: {result}")
    except Exception as e:
        print(f"Error reading the file: {e}")

# Menu to choose between image and video processing
while True:
    print("Choose an option:")
    print("1. Process Image")
    print("2. Process Video")
    print("3. Exit")

    choice = input("Enter your choice (1, 2, or 3): ")

    if choice == '1':
        image_path = input("Enter the path to the image: ")
        process_image(image_path)
    elif choice == '2':
        video_path = input("Enter the path to the video: ")
        process_video(video_path)
    elif choice == '3':
        print("Exiting the program.")
        break
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
