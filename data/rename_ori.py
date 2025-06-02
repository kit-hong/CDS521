import cv2
import os
from tqdm import tqdm


def rename_raw_files(path, foldername, ALLOWED_EXTENSIONS=('jpg', 'jpeg', 'png')):
    img_index = 0

    files = [f for f in os.listdir(path)
             if f.lower().endswith(ALLOWED_EXTENSIONS) and os.path.isfile(os.path.join(path, f))]

    temp_mapping = []

    for file in files:
        file_path = os.path.join(path, file)
        img = cv2.imread(file_path)

        if img is None:
            print(f"Failed to read image: {file_path}")
            continue

        temp_name = f"temp_{img_index}.jpg"
        temp_path = os.path.join(path, temp_name)
        temp_mapping.append((file_path, temp_path))
        img_index += 1

    for orig, temp in temp_mapping:
        try:
            os.rename(orig, temp)
        except Exception as e:
            print(f"Temporary rename failed for {orig}. Error: {e}")

    for idx, (_, temp_path) in enumerate(temp_mapping):
        final_name = f"{idx}.jpg"
        final_path = os.path.join(path, final_name)
        try:
            os.rename(temp_path, final_path)
        except Exception as e:
            print(f"Final rename failed for {temp_path}. Error: {e}")

    print(f"Processed - {img_index} images in {foldername}")

def main():
    main_folder = "raw_input"
        
    for foldername in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, foldername)
        if os.path.isdir(folder_path):
            rename_raw_files(folder_path,foldername)

if __name__ == "__main__":
    main()