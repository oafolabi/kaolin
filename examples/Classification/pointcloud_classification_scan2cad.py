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
from torch.utils.tensorboard import SummaryWriter
import kaolin.transforms as tfs
from utils import visualize_batch
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--run_number', type = str, default = '0', help='Run Number of log')
parser.add_argument('--tensorboard', type = int, default = 0, choices=[0,1],help='T/F To use Tensorboard or not')
#parser.add_argument('--modelnet-root', type=str, help='Root directory of the ModelNet dataset.')
#parser.add_argument('--categories', type=str, nargs='+', default=['chair', 'sofa'], help='list of object classes to use.')
parser.add_argument('--num-points', type=int, default=1024, help='Number of points to sample from meshes.')
parser.add_argument('--epochs', type=int, default=10, help='Number of train epochs.')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use.')

args = parser.parse_args()

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if(args.tensorboard == 0):
    args.tensorboard = False

else:
    args.tensorboard = True

transform = tfs.Compose([
    tfs.TriangleMeshToPointCloud(num_samples=args.num_points),
    tfs.NormalizePointCloud()
])

data_path = '/global/scratch/akashgokul/mined_scannet_chairs/data.csv'
data_frame = pd.read_csv(data_path)
data_frame.rename(columns={data_frame.columns[0]:'Filepath', data_frame.columns[1]:'ID'}, inplace=True)

train_dataset = Scan2CAD(data_frame,split='train',transform=transform, device=args.device)
train_loader = DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)

val_dataset = Scan2CAD(data_frame,split='validation',transform=transform, device=args.device)
val_loader = DataLoader(val_dataset,batch_size=1, shuffle=True)

test_dataset = Scan2CAD(data_frame,split='test',transform=transform, device=args.device)
test_loader = DataLoader(test_dataset,batch_size=1, shuffle=True)

#Same num_classes for all datasets
num_cad_classes = train_dataset.get_num_classes()
model = PointNetClassifier(num_classes=num_cad_classes).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

if(args.tensorboard):
    writer = SummaryWriter('runs/' + args.run_number)
else:
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
    if(args.tensorboard):
        writer.add_scalar('Training Loss', train_loss_e, e)
        writer.add_scalar('Training Accuracy', train_acc, e)
    else:
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
    if(args.tensorboard):
        writer.add_scalar('Validation Loss', val_loss_e, e)
        writer.add_scalar('Validation Accuracy', val_acc, e)
    else:
        val_loss_lst.append(val_loss_e)
        val_acc_lst.append(val_acc)
    
    test_loss = 0
    test_accuracy = 0 
    with torch.no_grad():
        print("\n Testing \n")
        for idx, batch in enumerate(tqdm(test_loader)):
            pred = model(batch[0])
            loss = criterion(pred, batch[1].view(-1))
            test_loss += loss.item()

            # Compute accuracy
            pred_label = torch.argmax(pred, dim=1)
            test_accuracy += torch.mean((pred_label == batch[1].view(-1)).float()).cpu().item()
            num_batches += 1

    test_loss_e = val_loss / num_batches
    test_acc = 100 * test_accuracy / num_batches
    print("-------\n"*2)
    print('Test accuracy:', test_acc)
    if(test_acc >= 80):
        np.save('train_acc', np.array(train_acc_lst))
        np.save('train_loss', np.array(train_loss_lst))
        np.save('val_acc', np.array(val_acc_lst))
        np.save('val_loss', np.array(val_loss_lst))

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
    
    if(args.tensorboard):
        writer.add_scalar('Final Test Accuracy', test_acc, 0)
    else:
        test_acc_lst.append(test_acc)
print('Test accuracy:', 100 * test_acc)

if(not args.tensorboard):
    plt.figure()
    plt.plot(np.array(train_acc_lst), label='train_acc', color = 'blue')
    plt.plot(np.array(val_acc_lst),label='val_acc', color = 'yellow')
    plt.plot(np.array(test_acc_lst), label='test_acc', color = 'red')
    plt.title("Train Acc vs Validation Acc")
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("Accuracy Percent")
    plt.savefig('run_' + args.run_number + "_accuracies_plot.png")

    plt.figure()
    plt.plot(np.array(train_loss_lst), label='train_loss', color = 'blue')
    plt.plot(np.array(val_loss_lst),label='val_loss', color = 'yellow')
    plt.title("Train Loss vs Validation Loss")
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.savefig('run_' + args.run_number + "_loss_plot.png")




# test_batch, labels = next(iter(test_loader))
# preds = model(test_batch)
# pred_labels = torch.max(preds, axis=1)[1]

# visualize_batch(test_batch, pred_labels, labels, args.categories)
