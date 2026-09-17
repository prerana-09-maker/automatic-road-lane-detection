import cv2
import numpy as np
import argparse
import os


def resize_image(image, width=1280, height=720):
    """
    Resize image to a fixed resolution.
    """
    return cv2.resize(image, (width, height))


def preprocess_image(image):
    """
    Convert image to grayscale, apply Gaussian blur,
    and perform Canny edge detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    return edges


def region_of_interest(edges):
    """
    Keep only the road region where lane lines
    are expected to appear.
    """

    height, width = edges.shape

    mask = np.zeros_like(edges)

    # Region of interest polygon
    polygon = np.array([
        [
            (int(width * 0.10), height),
            (int(width * 0.45), int(height * 0.60)),
            (int(width * 0.55), int(height * 0.60)),
            (int(width * 0.90), height)
        ]
    ], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)

    masked_edges = cv2.bitwise_and(
        edges,
        mask
    )

    return masked_edges


def detect_lines(cropped_edges):
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )

    return lines


def calculate_lane_line(lines, image_shape):
    height, width = image_shape[:2]

    left_lines = []
    right_lines = []

    if lines is None:
        return None, None

    for line in lines:
        # HoughLinesP normally returns [[x1, y1, x2, y2]]
        if np.asarray(line).size != 4:
            continue

        x1, y1, x2, y2 = np.asarray(line).reshape(-1)

        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if abs(slope) < 0.5:
            continue

        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    def make_line(line_data):
        if not line_data:
            return None

        slope, intercept = np.mean(line_data, axis=0)

        y1 = height
        y2 = int(height * 0.60)

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return (x1, y1, x2, y2)

    left_line = make_line(left_lines)
    right_line = make_line(right_lines)

    return left_line, right_line


def draw_lane_lines(image, left_line, right_line):
    overlay = image.copy()

    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 8)

    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 8)

    result = cv2.addWeighted(image, 0.8, overlay, 1.0, 0)

    return result

def process_frame(frame):
    frame = resize_image(frame)

    edges = preprocess_image(frame)
    cropped_edges = region_of_interest(edges)

    lines = detect_lines(cropped_edges)

    left_line, right_line = calculate_lane_line(lines, frame.shape)

    result = draw_lane_lines(frame, left_line, right_line)

    return result


def process_image(input_path, output_path):
    """
    Process a single image.
    """

    image = cv2.imread(input_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {input_path}"
        )

    result = process_frame(image)

    # Create output directory if required
    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    success = cv2.imwrite(
        output_path,
        result
    )

    if not success:
        raise IOError(
            f"Could not save output image: {output_path}"
        )

    print("Image processing completed.")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")


def process_video(input_path, output_path):
    """
    Process a video frame by frame.
    """

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video: {input_path}"
        )

    width = 1280
    height = 720

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    # Some videos may not report FPS correctly
    if fps <= 0:
        fps = 30

    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    # MP4 output
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()

        raise IOError(
            f"Could not create output video: {output_path}"
        )

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = process_frame(frame)

        writer.write(result)

        frame_count += 1

    cap.release()
    writer.release()

    print("Video processing completed.")
    print(f"Input       : {input_path}")
    print(f"Output      : {output_path}")
    print(f"Frames      : {frame_count}")
    print(f"FPS         : {fps:.2f}")


def is_video_file(path):
    """
    Check whether the input file is a video.
    """

    video_extensions = [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv"
    ]

    extension = os.path.splitext(
        path
    )[1].lower()

    return extension in video_extensions


def main():

    parser = argparse.ArgumentParser(
        description="Automatic Road Lane Detection using Computer Vision"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input image or video"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the processed output"
    )

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        print(
            f"Error: Input file does not exist: {input_path}"
        )
        return

    try:

        if is_video_file(input_path):

            process_video(
                input_path,
                output_path
            )

        else:

            process_image(
                input_path,
                output_path
            )

    except Exception as error:

        print(
            f"Error: {error}"
        )


if __name__ == "__main__":
    main()