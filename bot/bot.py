# ------------------------------------------------
#   Old School Runescape CNN BOT
#
#       Author: Matheus Teixeira
# 
#       Email: matheustalves@outlook.com
# ------------------------------------------------

import cv2
import numpy as np
from tqdm import tqdm

import torch

from build_dataset import Dataset_Builder
from net import Net
from train import train

from mss import mss
from pynput.mouse import Button, Controller
import time

device = torch.device("cuda:0")

# build .npy dataset from /dataset/ samples
REBUILD_DATASET = False 
if REBUILD_DATASET:
    builder = Dataset_Builder()
    builder.build()


network = Net()
train(network)

# identity tensor for checking if network identified a monster
hit = torch.Tensor(np.eye(2)[1]) 
hit = torch.argmax(hit)

mouse = Controller()

bbox = {'top': 50, 'left': 30, 'width': 500, 'height': 300} # Game Canvas Size
sct = mss() # MSS will grab the game canvas of bbox size

# loads template for checking if character is already attacking
template2 = cv2.imread('../static/monster_hp.jpg', 0)
w2, h2 = template2.shape[::-1]

cX = 0
cY = 0

# Execution time
t_end = time.time() + 60*0.2
while time.time() < t_end:

    # grabs game canvas and applies grayscale
    sct_img = sct.grab(bbox)
    output = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGR2GRAY)

    # slices game canvas in 6 rows / 10 columns
    start_row, start_col = int(0), int(0)
    img_slices = []
    for row in range(0, 6):
        end_row = int(50*(1+row))
        for col in range(0, 10):
            end_col = int(50*(1+col))
            cropped = output[start_row:end_row, start_col:end_col]
            img_slices.append(cropped)
            start_col = end_col
        start_row = end_row
        start_col = int(0)

    # image slices to Tensors and normalization
    img_slices = torch.Tensor([i for i in img_slices]).view(-1, 50, 50)
    img_slices = img_slices/255.0

    monster_identified = False
    with torch.no_grad(): # reset gradients from network
        for i in tqdm(range(len(img_slices))): # search for monster in each canvas slice
            net_out = network(img_slices[i].view(-1, 1, 50, 50))[0]
            predicted_class = torch.argmax(net_out)

            if predicted_class == hit:
                monster_identified = True
                if i < 10:
                    cv2.rectangle(output, (50*i, 0),
                                  (50*(i+1), 50), (0, 255, 0), 2)
                    cX = 25*(i+1)
                    cY = 25
                elif i < 20:
                    cv2.rectangle(output, (50*(i-10), 50),
                                  (50*(i-9), 100), (0, 255, 0), 2)
                    cX = 25*(i-9)
                    cY = 50
                elif i < 30:
                    cv2.rectangle(output, (50*(i-20), 100),
                                  (50*(i-19), 150), (0, 255, 0), 2)
                    cX = 25*(i-19)
                    cY = 75
                elif i < 40:
                    cv2.rectangle(output, (50*(i-30), 150),
                                  (50*(i-29), 200), (0, 255, 0), 2)
                    cX = 25*(i-29)
                    cY = 100
                elif i < 50:
                    cv2.rectangle(output, (50*(i-40), 200),
                                  (50*(i-39), 250), (0, 255, 0), 2)
                    cX = 25*(i-39)
                    cY = 125
                else:
                    cv2.rectangle(output, (50*(i-50), 250),
                                  (50*(i-49), 300), (0, 255, 0), 2)
                    cX = 25*(i-49)
                    cY = 150

    res2 = cv2.matchTemplate(output, template2, cv2.TM_CCOEFF_NORMED)
    threshold2 = 0.69
    loc2 = np.where(res2 >= threshold2)

    in_combat = False
    for pt in zip(*loc2[::-1]):
        if pt is not None:
            in_combat = True
        cv2.rectangle(output, pt, (pt[0]+w2, pt[1]+h2), (0, 255, 255), 2)

    if in_combat:
        print("In combat.")
    else:
        print("Outside combat.")
        if monster_identified:
            mouse.position = (cX, cY)
            mouse.click(Button.left, 1)

    cv2.imshow('screen', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break