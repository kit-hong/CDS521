import cv2
import os

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

def rotate_image(image, angles=[0, 15, 30, 45, 60, 75, 90]):
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
    cv2.imwrite(output_path, resized_img)

def process(path, label, label2idx, outdir):
    img_index = 0
    idx = label2idx[label]

    for file in os.listdir(path):
        file_path = os.path.join(path,file)
        if file_path.endswith('.jpg') and os.path.isfile(file_path):
            try:
                img = cv2.imread(file_path)

                resize_rename_write(img, idx, img_index, outdir, "original")
                img_index += 1

                flipped_ori_imgs = flip(img)
                for flipped_ori_img in flipped_ori_imgs:
                    resize_rename_write(flipped_ori_img, idx, img_index, outdir, "original_flipped")
                    img_index += 1

                rotated_imgs = rotate_image(img)
                for rotated_img in rotated_imgs:
                    resize_rename_write(rotated_img, idx, img_index, outdir, "original_rotated")
                    img_index += 1

                    flipped_imgs = flip(rotated_img)
                    for flipped_img in flipped_imgs:
                        resize_rename_write(flipped_img, idx, img_index, outdir, "rotated_flipped") 
                        img_index += 1

            except Exception as e:
                print(f"Failed at - {file}. Error: {e}")

def main():
    main_folder = "/raw_input"
    outdir = "/output"
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