OpenCV Sudoku Solver

This project is an OpenCV-based Sudoku solver that uses image processing and deep learning to identify and solve Sudoku puzzles. The application leverages OpenCV for preprocessing the images and extracting the Sudoku grid, and a Convolutional Neural Network (CNN) model for recognizing the digits.

Key Features:

Image Preprocessing:

Uses Gaussian blur and adaptive thresholding to preprocess the input image.
Detects and extracts the Sudoku grid using contour detection.
Applies perspective transformation to isolate the Sudoku grid.
Digit Recognition:

Utilizes a trained CNN model to recognize and predict the digits in the Sudoku grid.
Processes and centers the detected digits within the grid cells for accurate prediction.
Sudoku Solver:

Implements an efficient backtracking algorithm to solve the Sudoku puzzle.
Uses exact cover and the Dancing Links technique to handle the constraint satisfaction problem.
Result Display:

Overlays the solved Sudoku digits onto the original image.
Applies inverse perspective transformation to map the solved digits back onto the original image frame.
Modules and Functions:

solve_sudoku(size, grid): Solves the Sudoku puzzle using the backtracking algorithm.
check_contour(img): Preprocesses the image and extracts the Sudoku grid contour.
predict(img, model): Predicts the digits in the Sudoku grid using the trained CNN model.
inv_transformation(mask, img, predicted_matrix, solved_matrix, corners): Overlays the solved digits onto the original image.
Image Processing Functions: Includes functions like preprocess, extract_frame, extract_numbers, center_numbers, and more for handling image transformations and digit extraction.
Usage:

Capture or load an image of a Sudoku puzzle.
Use the check_contour function to detect and extract the Sudoku grid.
Predict the digits using the predict function with a trained CNN model.
Solve the Sudoku puzzle using the solve_wrapper function.
Display the solved puzzle on the original image using inv_transformation.
Dependencies:

OpenCV
NumPy
Keras/TensorFlow
cvzone
sklearn
Installation:

Clone the repository.
Install the required dependencies using pip install -r requirements.txt.
Run the main script to start the Sudoku solver.
