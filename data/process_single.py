import cv2
import os
import numpy as np
from tqdm import tqdm
from .data_preprocessing import create_mapping, process

def main(idx, labels = ["Unripe", "Early Ripening", "Ripe", "Fully Ripe", "Overripe"]):
    label2idx, idx2label = create_mapping(labels)
    main_folder = os.path.join("data/raw_input",labels[idx])
    outdir = os.path.join("data/processed_data",labels[idx])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    process(main_folder,labels[idx],label2idx,outdir)

if __name__ == "__main__":
    main(1)