import os
import numpy as np
import cv2
from tqdm import tqdm


class Dataset_Builder():
    IMG_SIZE = 50
    SPIDERS = "Dataset/Spides"
    NOT_SPIDERS = "Dataset/Not_Spiders"
    LABELS = {NOT_SPIDERS: 0, SPIDERS: 1}
    training_data = []
    spidercount = 0
    not_spiderscount = 0

    def build(self):
        for label in self.LABELS:
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data.append(
                        [np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.SPIDERS:
                        self.spidercount += 1
                    elif label == self.NOT_SPIDERS:
                        self.not_spiderscount += 1

                except Exception as e:
                    pass

        np.random.shuffle(self.training_data)
        np.save("dataset.npy", self.training_data)
        print("Spiders:", self.spidercount)
        print("Not Spiders:", self.not_spiderscount)
