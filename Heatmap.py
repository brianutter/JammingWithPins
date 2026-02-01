## Import necessary libraries
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray

## Define input and output paths
file_path = Path(r"C:\Users\User\Downloads\Documents\School\Research\Run2-07-27-340f-IP-FW\W-Polarizer")
output_path = Path(r"C:\Users\User\Downloads\Documents\School\Research\Results")
output_path.mkdir(parents=True, exist_ok=True)

## Configuration parameters
rotate_deg = 0
crop_rows = None   # replace with: slice(start_row, end_row)
crop_cols = None   # replace with: slice(start_col, end_col)
threshold = 0  # pixel intensity threshold (0-255); 0 means no thresholding
framerate = 30

## Define helper functions
def sort_input(file_name):
    '''Sorts input files in numerical order'''
    nums = re.findall(r"\d+", file_name.stem)
    return int(nums[-1]) if nums else 1000000

def grayscale_image(image_path):
    '''Read and convert image to grayscale'''
    img = io.imread(str(image_path))
    if img.ndim == 3:
        img = rgb2gray(img) # Convert RGB to grayscale from 0 to 1
        img = (img * 255).astype(np.uint8) # Convert to 0-255 uint8
    else:
        img = img.astype(np.uint8)
    return img

def process_image(image):
    '''Process image: rotate, crop, threshold'''
    if rotate_deg != 0:
        image = rotate(
            image,
            rotate_deg,
            resize=False,
            preserve_range=True
        ).astype(np.uint8)

    if crop_rows is not None:
        image = image[crop_rows, :]

    if crop_cols is not None:
        image = image[:, crop_cols]

    if threshold > 0:
        mask = image > threshold
        image = (image * mask).astype(np.uint8)

    return image

## Main execution block
if __name__ == "__main__":

    # Prepare output directory
    run_name = file_path.parts[-2]
    output_dir = output_path / run_name
    output_dir.mkdir(exist_ok=True)

    # Gather and sort image files
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = sorted(
        [f for f in file_path.iterdir() if f.suffix.lower() in extensions],
        key=sort_input
    )

    # Initialize variables for animation
    frames = []
    sum_image = None
    count = 0
    print("Initialized. Processing images...")

    # Process each image
    for i in images:
        img = grayscale_image(i)
        img = process_image(img)

        if sum_image is None:
            sum_image = np.zeros(img.shape, dtype=np.float64)

        frames.append(img)
        sum_image += img.astype(np.float64)
        count += 1
        print(f"Processed image: {i.name}")

    # Compute average image
    avg_image = sum_image / count

    # Name the output files 
    video_file = output_dir / f"Heatmap_{run_name}.mp4"
    avg_file = output_dir / f"Average_{run_name}.png"
    avg_csv_file = output_dir / f"Average_{run_name}.csv"
    
    # Create and save animation
    h, w = frames[0].shape
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = plt.axes([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(frames[0], cmap="plasma")

    def update(i): # update function for animation
        im.set_array(frames[i])
        print(f"Rendering frame {i + 1} / {len(frames)}")
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000 / framerate, blit=True
    )

    writer = animation.FFMpegWriter(fps=framerate)
    ani.save(str(video_file), writer=writer)
    plt.close(fig)

    # Save average image as PNG
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = plt.axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(avg_image, cmap="plasma", aspect="auto")
    plt.savefig(avg_file, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


    # Save average image as CSV
    np.savetxt(str(avg_csv_file), avg_image, delimiter=",")

    print("Files saved")
