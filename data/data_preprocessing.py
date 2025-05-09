import cv2
import os
from tqdm import tqdm
import multiprocessing as mp

''''
/project-folder
│
├── data_preprocessing.py    
│
├── /raw_input                   
│   ├── /Unripe              
│   ├── /Early Ripening      
│   ├── /Ripe                
│   ├── /Fully Ripe         
│   └── /Overripe            
│
└── /output                  
'''

def create_mapping(labels):
    label2idx = {label:idx for idx,label in enumerate(labels)}
    idx2label = {idx:label for idx,label in enumerate(labels)}
    return label2idx, idx2label

def rotate_image(image, angles=[15, 30, 45, 60, 75, 90]):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotated_images = []
    
    for a in angles:
        rotation_matrix = cv2.getRotationMatrix2D(center, a, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
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

def resize_rename_write(img, idx, img_index, outdir, tag):
    resized_img = cv2.resize(img, (224, 224))
    output_name = f"{idx}_{img_index}_{tag}.jpg"
    output_path = os.path.join(outdir, output_name)
    cv2.imwrite(output_path, resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

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

        if img is None:
            print(f"Failed to read image: {file_path}")
            continue
        try:
            resize_rename_write(img, idx, img_index, label_outdir, "original")
            img_index += 1

            flipped_ori_imgs = flip(img)
            for flipped_ori_img in tqdm(flipped_ori_imgs, desc="Flipping Original", leave=False, position=1):
                resize_rename_write(flipped_ori_img, idx, img_index, label_outdir, "original_flipped")
                img_index += 1

            rotated_imgs = rotate_image(img)
            for rotated_img in tqdm(rotated_imgs, desc="Rotating Original", leave=False, position=2):
                resize_rename_write(rotated_img, idx, img_index, label_outdir, "original_rotated")
                img_index += 1

                flipped_imgs = flip(rotated_img)
                for flipped_img in tqdm(flipped_imgs, desc="Flipping Rotated", leave=False, position=3):
                    resize_rename_write(flipped_img, idx, img_index, label_outdir, "rotated_flipped") 
                    img_index += 1

        except Exception as e:
            print(f"Failed at - {file}. Error: {e}")

def main():
    main_folder = "raw_input"
    outdir = "output"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    labels = ["Unripe", "Early Ripening", "Ripe", "Fully Ripe", "Overripe"]
    label2idx, idx2label = create_mapping(labels)

    pool = mp.Pool(processes=min(mp.cpu_count(), len(labels)))
    tasks = []

    for foldername in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, foldername)
        if os.path.isdir(folder_path):
            tasks.append((folder_path, foldername, label2idx, outdir))

    pool.starmap(process, tasks)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()