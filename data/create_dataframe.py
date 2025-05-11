import os
import pandas as pd
from tqdm import tqdm

def create_mapping(labels):
    label2idx = {label:idx for idx,label in enumerate(labels)}
    idx2label = {idx:label for idx,label in enumerate(labels)}
    return label2idx, idx2label

def create_dataframe(path, labels, label2idx, ALLOWED_EXTENSIONS = ('jpg', 'jpeg', 'png')):
  data = []

  for foldername in tqdm(os.listdir(path), desc="Creating DataFrame", position=0):
    folderpath = os.path.join(path,foldername)

    if foldername in labels and os.path.isdir(folderpath):

      data.extend(
          # (imagepaths, labels)
          (os.path.join(folderpath,image), label2idx[foldername])
          for image in tqdm(os.listdir(folderpath), desc=f"Processing {foldername}", position=1, leave=False)
          if os.path.isfile(os.path.join(folderpath,image)) and image.lower().endswith(ALLOWED_EXTENSIONS)
      )

  df = pd.DataFrame(data, columns = ['imagepath', 'label'])
  return df

def main():
    labels = ["Unripe", "Early Ripening", "Ripe", "Fully Ripe", "Overripe"]
    label2idx, idx2label = create_mapping(labels)

    data_path = "data/output"
    
    df = create_dataframe(data_path, labels, label2idx)
    df.to_csv(os.path.join(os.getcwd(), 'data.csv'), index=False)

if __name__ == "__main__":
    main()