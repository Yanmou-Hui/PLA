import os
from PIL import Image
from tqdm import tqdm


def check_images_in_directory(directory):
    """Check all images in a directory and its subdirectories.

    Args:
        directory (str): Path to the directory.
    """
    valid_image_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', 'PNG', 'JPG', 'JPEG', 'BMP', 'GIF', 'TIFF')
    for root, dirs, files in os.walk(directory):
        for file in tqdm(files, desc=root):
            if file.lower().endswith(valid_image_formats):
                file_path = os.path.join(root, file)
                try:
                    Image.open(file_path).convert('RGB')
                except (OSError, ValueError) as e:
                    print(f"Invalid image: {file_path} - {e}")


# Example usage:
if __name__ == '__main__':
    check_images_in_directory('../dataset/genimage')
