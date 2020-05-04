import argparse

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
sys.path.append("..")
from kaolin.datasets import ModelNet, Scan2CAD
from kaolin.models.PointNet import PointNetClassifier
import kaolin.transforms as tfs
# from utils import visualize_batch
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--run_number', type = str, default = '0', help='Run Number of log')
#parser.add_argument('--modelnet-root', type=str, help='Root directory of the ModelNet dataset.')
#parser.add_argument('--categories', type=str, nargs='+', default=['chair', 'sofa'], help='list of object classes to use.')
parser.add_argument('--num-points', type=int, default=1024, help='Number of points to sample from meshes.')
parser.add_argument('--epochs', type=int, default=10, help='Number of train epochs.')
parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use.')

args = parser.parse_args()

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = tfs.Compose([
    tfs.TriangleMeshToPointCloud(num_samples=args.num_points),
    tfs.NormalizePointCloud()
])

data_path = '/global/scratch/oafolabi/data/mined_scannet_chairs/data.csv'
data_frame = pd.read_csv(data_path)
data_frame.rename(columns={data_frame.columns[0]:'Filepath', data_frame.columns[1]:'ID'}, inplace=True)

train_dataset = Scan2CAD(data_frame,split='train',transform=transform, device=args.device)
print(train_dataset.get_num_classes())
train_loader = DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)

val_dataset = Scan2CAD(data_frame,split='validation',transform=transform, device=args.device)
print(len(val_dataset))
val_loader = DataLoader(val_dataset,batch_size=1, shuffle=True)

test_dataset = Scan2CAD(data_frame,split='test',transform=transform, device=args.device)
test_loader = DataLoader(test_dataset,batch_size=1, shuffle=True)
print(len(test_dataset))

# true_test_dataset = Scan2CAD(true_df, split='train-tl',transform=transform, device=args.device)
# true_test_loader = DataLoader(true_test_dataset, batch_size = 1, shuffle = True)
# test_loader = true_test_loader
#Same num_classes for all datasets
#316 Classes

num_cad_classes = train_dataset.get_num_classes()
model = PointNetClassifier(num_classes=num_cad_classes)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = torch.nn.DataParallel(model)
else:
    model.to(args.device)


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
    

    test_accuracy = val_acc 
    num_batches = 0

    with torch.no_grad():
        print("\n Testing \n")
        for idx, batch in enumerate(tqdm(test_loader)):
            pred = model(batch[0])

            # Compute accuracy
            pred_label = torch.argmax(pred, dim=1)
            test_accuracy += torch.mean((pred_label == batch[1].view(-1)).float()).cpu().item()
            num_batches += 1

    test_acc = test_accuracy / num_batches

    print("-------\n"*2)
    print('Test accuracy:', 100 * test_acc)
    if(100 * test_acc >= 80):
        # np.save('train_acc_' + args.run_number, np.array(train_acc_lst))
        # np.save('train_loss_' + args.run_number, np.array(train_loss_lst))
        # np.save('val_acc_' + args.run_number, np.array(val_acc_lst))
        # np.save('val_loss_' + args.run_number, np.array(val_loss_lst))
        # torch.save(model, 'pointnet_model_' + args.run_number + '.pt')
        # torch.save(model.state_dict(), 'pointnet_model_state_dict' + args.run_number + '.pt')
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

np.save('train_acc_full_'+args.run_number, np.array(train_acc_lst))
np.save('train_loss_full_'+args.run_number, np.array(train_loss_lst))
np.save('val_acc_full_'+args.run_number, np.array(val_acc_lst))
np.save('val_loss_full_'+args.run_number, np.array(val_loss_lst))




# test_batch, labels = next(iter(test_loader))
# preds = model(test_batch)
# pred_labels = torch.max(preds, axis=1)[1]

# visualize_batch(test_batch, pred_labels, labels, args.categories)
