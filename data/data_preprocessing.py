import cv2
import os
import numpy as np
from tqdm import tqdm

def create_mapping(labels):
    label2idx = {label:idx for idx,label in enumerate(labels)}
    idx2label = {idx:label for idx,label in enumerate(labels)}
    return label2idx, idx2label

def rotate_image(image, angles=np.arange(0, 360, 30)):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotated_images = []
    
    for a in angles:
        rotation_matrix = cv2.getRotationMatrix2D(center, a, 1.0)
        # Fix black corners by reflecting the border
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        rotated_images.append(rotated_image)
    
    return rotated_images

def horizontal_flip(image):
    return cv2.flip(image, 1)  

def vertical_flip(image):
    return cv2.flip(image, 0)  

def flip(image):
    flipped_images = []
    flipped_images.append(horizontal_flip(image))
    flipped_images.append(vertical_flip(image))
    return flipped_images

def shear(image, factors=[-0.1, 0.1]):
    h, w = image.shape[:2]
    sheared_images = []
    for factor in factors:
        M_x = np.float32([[1, factor, 0], [0, 1, 0]])
        M_y = np.float32([[1, 0, 0], [factor, 1, 0]])
        sheared_images.append(cv2.warpAffine(image, M_x, (w, h), borderMode=cv2.BORDER_REFLECT))
        sheared_images.append(cv2.warpAffine(image, M_y, (w, h), borderMode=cv2.BORDER_REFLECT))
    return sheared_images

def resize_rename_write(img, idx, img_index, outdir, ori_id):
    desired_size = 224
    h, w = img.shape[:2]

    # Scale the image proportionally - avoid distortion
    scale = desired_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to make it square (224x224) - avoid distortion
    delta_w = desired_size - new_w
    delta_h = desired_size - new_h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT)

    # Step 3: Save the final 224x224 image
    output_name = f"{idx}_{img_index}_{ori_id}.jpg"
    output_path = os.path.join(outdir, output_name)
    cv2.imwrite(output_path, padded_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

def process(path, label, label2idx, outdir, ALLOWED_EXTENSIONS = ('jpg', 'jpeg', 'png')):
    img_index = 0
    idx = label2idx[label]

    label_outdir = os.path.join(outdir, label)
    if not os.path.exists(label_outdir):
        os.makedirs(label_outdir)

    files = [f for f in os.listdir(path) if f.lower().endswith(ALLOWED_EXTENSIONS) and os.path.isfile(os.path.join(path, f))]

    for file in tqdm(files, desc=f"Processing {label}", position=0):
        file_path = os.path.join(path,file)
        img = cv2.imread(file_path)
        ori_id = os.path.splitext(file)[0].replace(" ", "_")

        if img is None:
            print(f"Failed to read image: {file_path}")
            continue
        try:
            resize_rename_write(img, idx, img_index, label_outdir, ori_id)
            img_index += 1
            
            # flipping
            flipped_imgs = flip(img)
            for flipped_img in flipped_imgs:
                resize_rename_write(flipped_img, idx, img_index, label_outdir, ori_id)
                img_index += 1

            # rotation
            rotated_imgs = rotate_image(img)
            for rotated_img in rotated_imgs:
                resize_rename_write(rotated_img, idx, img_index, label_outdir, ori_id)
                img_index += 1

            # shearing
            sheared_imgs = shear(img)
            for sheared_img in sheared_imgs:
                resize_rename_write(sheared_img, idx, img_index, label_outdir, ori_id)
                img_index += 1

        except Exception as e:
            print(f"Failed at - {file}. Error: {e}")
    
    print(f"Processed - {img_index} images")


def main():
    main_folder = "raw_input"
    outdir = "processed_data"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    labels = ["Unripe", "Early Ripening", "Ripe", "Fully Ripe", "Overripe"]
    label2idx, idx2label = create_mapping(labels)

    for foldername in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, foldername)
        if os.path.isdir(folder_path):
            process(folder_path,foldername,label2idx,outdir)

if __name__ == "__main__":
    main()