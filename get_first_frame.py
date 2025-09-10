import cv2
import os
import sys

def get_first_frame(video_path):
    """
    Extract the first frame from a video and save it as a PNG file.
    
    Args:
        video_path (str): Path to the input video file
        
    Returns:
        str: Path to the saved PNG file, or None if failed
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return None
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return None
    
    # Read the first frame
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Could not read first frame from '{video_path}'.")
        cap.release()
        return None
    
    # Generate output filename (same name as video but with .png extension)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"{video_name}.png"
    
    # Save the first frame as PNG
    success = cv2.imwrite(output_path, frame)
    
    # Release the video capture
    cap.release()
    
    if success:
        print(f"Successfully saved first frame as '{output_path}'")
        return output_path
    else:
        print(f"Error: Failed to save frame as '{output_path}'")
        return None

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python get_first_frame.py <video_file>")
        print("Example: python get_first_frame.py agent1_image.mp4")
        sys.exit(1)
    
    video_file = sys.argv[1]
    result = get_first_frame(video_file)
    
    if result is None:
        sys.exit(1)

if __name__ == "__main__":
    main()
