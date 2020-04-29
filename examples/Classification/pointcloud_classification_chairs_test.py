
###
# File to Test the saved Pointnet V1 model for the mined_scannet_chairs dataset
###

import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
sys.path.append("..")
from kaolin.datasets import ModelNet, Scan2CAD
from kaolin.models.PointNet import PointNetClassifier
import kaolin.transforms as tfs
# from utils import visualize_batch
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, help = 'Path to CSV containing Filepaths - CAD IDs')

args = parser.parse_args()

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = tfs.Compose([
    tfs.TriangleMeshToPointCloud(num_samples=1024),
    tfs.NormalizePointCloud()
])

data_path = args.csv_path
data_frame = pd.read_csv(data_path)
data_frame.rename(columns={data_frame.columns[0]:'Filepath', data_frame.columns[1]:'ID'}, inplace=True)

#Instantiates dataset and dataloader for the data from the csv file
dataset = Scan2CAD(data_frame, split='none', transform=transform, device = args.device)
dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)


# From original data there are 316 classes
model = PointNetClassifier(num_classes=316).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

train_acc_lst = []
train_loss_lst = []
val_loss_lst = []
val_acc_lst = []
test_acc_lst = []

for e in range(args.epochs):

    print('###################')
    print('Epoch:', e)
    print('###################')

    train_loss = 0.
    train_accuracy = 0.
    num_batches = 0

    model.train()

    optimizer.zero_grad()
    for idx, batch in enumerate(tqdm(train_loader)):
        pred = model(batch[0])
        loss = criterion(pred, batch[1].view(-1))
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        pred_label = torch.argmax(pred, dim=1)
        train_accuracy += torch.mean((pred_label == batch[1].view(-1)).float()).detach().cpu().item()
        num_batches += 1
    
    train_loss_e = train_loss / num_batches
    train_acc = train_accuracy / num_batches
    print('Train loss:', train_loss_e)
    print('Train accuracy:', 100 * train_acc)

    train_loss_lst.append(train_loss_e)
    train_acc_lst.append(train_acc)

    val_loss = 0.
    val_accuracy = 0.
    num_batches = 0

    model.eval()

    with torch.no_grad():
        print("\n Validation Testing \n")
        for idx, batch in enumerate(tqdm(val_loader)):
            pred = model(batch[0])
            loss = criterion(pred, batch[1].view(-1))
            val_loss += loss.item()

            # Compute accuracy
            pred_label = torch.argmax(pred, dim=1)
            val_accuracy += torch.mean((pred_label == batch[1].view(-1)).float()).cpu().item()
            num_batches += 1

    val_loss_e = val_loss / num_batches
    val_acc = val_accuracy / num_batches
    print('Val loss:', val_loss_e)
    print('Val accuracy:', 100 * val_acc)

    val_loss_lst.append(val_loss_e)
    val_acc_lst.append(val_acc)
    

    test_accuracy = 0 
    num_batches = 0

    with torch.no_grad():
        print("\n Testing \n")
        for idx, batch in enumerate(tqdm(test_loader)):
            pred = model(batch[0])

            # Compute accuracy
            pred_label = torch.argmax(pred, dim=1)
            test_accuracy += torch.mean((pred_label == batch[1].view(-1)).float()).cpu().item()
            num_batches += 1

    test_acc = 100 * test_accuracy / num_batches
    print("-------\n"*2)
    print('Test accuracy:', test_acc)
    if(test_acc >= 90):
        np.save('train_acc', np.array(train_acc_lst))
        np.save('train_loss', np.array(train_loss_lst))
        np.save('val_acc', np.array(val_acc_lst))
        np.save('val_loss', np.array(val_loss_lst))
        break

# test_loader = DataLoader(ModelNet(args.modelnet_root, categories=args.categories,
#                                   split='test', transform=transform, device=args.device),
#                          shuffle=True, batch_size=15)


model.eval()

test_acc = 0.
print("Testing")

with torch.no_grad():
    num_batches = 0
    for idx, test_batch in enumerate(tqdm(test_loader)):
        pred = model(test_batch[0])
        pred_labels = torch.argmax(pred, dim=1)
        #assuming test-batch 1
        if(pred_labels[0].int() == test_batch[1][0].int()):
            test_acc += 1
        #test_acc += torch.mean((pred_labels == test_batch[1].view(-1)).float().cpu().item())
        num_batches += 1
    test_acc = test_acc / num_batches
    

print('Test accuracy:', 100 * test_acc)
torch.save(model.state_dict(), "mined_scannet_chairs_pointnet1_model.pt")




# test_batch, labels = next(iter(test_loader))
# preds = model(test_batch)
# pred_labels = torch.max(preds, axis=1)[1]

# visualize_batch(test_batch, pred_labels, labels, args.categories)
