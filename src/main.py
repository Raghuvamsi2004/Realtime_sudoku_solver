import cv2
from capture import capture_video
from preprocessing import preprocess_image, warp_perspective
from digit_recognition import predict_digit
from sudoku_solver import solve_sudoku


def main():
    # Start video capture
    video_capture = capture_video()

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Preprocess the image for Sudoku detection
        preprocessed_image = preprocess_image(frame)

        # Warp the perspective to get a top-down view of the Sudoku grid
        warped_image = warp_perspective(preprocessed_image)

        # Predict digits from the warped image
        digits = predict_digit(warped_image)

        # Solve the Sudoku puzzle
        solved_grid = solve_sudoku(digits)

        # Draw the solved Sudoku grid on the original frame
        annotated_frame = overlay_solution(frame, solved_grid)

        # Display the resulting frame
        cv2.imshow('Real-Time Sudoku Solver', annotated_frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()


def overlay_solution(frame, solved_grid):
    """
    Overlay the solved Sudoku grid on the original frame.
    :param frame: Original video frame
    :param solved_grid: 9x9 list of solved Sudoku values
    :return: Frame with the Sudoku solution overlaid
    """
    # You will need to implement this function based on how you want to draw the solution.
    # For example, you might draw the grid lines and the solved numbers.

    # Example implementation (customize as needed):
    grid_size = 9
    cell_size = frame.shape[0] // grid_size

    # Draw grid lines
    for i in range(1, grid_size):
        cv2.line(frame, (0, i * cell_size), (frame.shape[1], i * cell_size), (0, 255, 0), 2)
        cv2.line(frame, (i * cell_size, 0), (i * cell_size, frame.shape[0]), (0, 255, 0), 2)

    # Draw the solved digits
    for row in range(grid_size):
        for col in range(grid_size):
            digit = solved_grid[row][col]
            if digit != 0:
                text = str(digit)
                position = (col * cell_size + cell_size // 4, row * cell_size + cell_size // 2)
                cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame


if __name__ == "__main__":
    main()

