import os
import numpy as np
import cv2
import random

class DataPreprocessor():
    def __init__(self):
        self.classes = {
            "Angry": 0, 
            "Fear": 1, 
            "Happy": 2, 
            "Sad": 3, 
            "Surprise": 4, 
            "Neutral": 5
        }
        self.img_size = 48
        self.target_count = 7500 

    def augment_image(self, img):
        """Generates a random variation of the image"""
        rows, cols = img.shape
        
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        img = cv2.warpAffine(img, M, (cols, rows))

        alpha = random.uniform(0.8, 1.2) # Contrast
        beta = random.uniform(-20, 20)   # Brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        return img

    def standardise(self, podaci):
        return (podaci.astype('float32') / 127.5) - 1
    
    def shuffle(self, oznake, podaci):
        assert len(oznake) == len(podaci)
        p = np.random.permutation(len(oznake))
        return (oznake[p], podaci[p])
    
    def load_from_dir(self, root_path):
        data_list = []
        label_list = []

        print(f"Loading and balancing data from {root_path}...")

        for emotion, label_id in self.classes.items():
            path = os.path.join(root_path, emotion)
            if not os.path.exists(path):
                print(f"Skipping {emotion} (Folder not found)")
                continue
                
            class_originals = []
            for img_name in os.listdir(path):
                try:
                    img_path = os.path.join(path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        class_originals.append(img)
                except Exception as e:
                    continue
            
            for img in class_originals:
                data_list.append(img)
                label_list.append(label_id)

            current_count = len(class_originals)
            print(f"  {emotion}: Loaded {current_count} originals.")

            if current_count < self.target_count:
                needed = self.target_count - current_count
                print(f"    -> Augmenting {needed} more images...")
                
                for _ in range(needed):
                    base_img = random.choice(class_originals)
                    aug_img = self.augment_image(base_img)
                    
                    data_list.append(aug_img)
                    label_list.append(label_id)
            
            print(f"    -> Final count for {emotion}: {len(data_list) - (len(data_list) - self.target_count if len(data_list) > self.target_count else 0)}")

        oznake = np.array(label_list)
        podaci = np.array(data_list)
        
        podaci = podaci.reshape((-1, self.img_size, self.img_size))
        
        print(f"Total processed images: {len(oznake)}")
        return (oznake, podaci)
    
    def oneHot(self, oznake):
        num_classes = 6 
        nove_oznake = []
        for i in range(len(oznake)):
            temp = np.zeros(num_classes)
            temp[oznake[i]] = 1
            nove_oznake.append(temp)
        return np.array(nove_oznake)

    def save(self, savelabel, savedata, label, data):
        np.save(savelabel, label)
        np.save(savedata, data)
        print(f"Saved dataset to {savelabel} and {savedata}")

    def process(self, dir_path, savelabel, savedata):
        labels, data = self.load_from_dir(dir_path)
        labels, data = self.shuffle(labels, data)
        data = self.standardise(data)
        self.save(savelabel, savedata, self.oneHot(labels), data)

if __name__ == '__main__':
    predprocesor = DataPreprocessor()
    predprocesor.process("./dataset/train", "train_labels.npy", "train_data.npy")