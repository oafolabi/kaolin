import argparse

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append("..")
from kaolin.datasets import ModelNet, Scan2CAD
from kaolin.models.PointNet import PointNetClassifier
import kaolin.transforms as tfs
from utils import visualize_batch
import pandas as pd

parser = argparse.ArgumentParser()
#parser.add_argument('--modelnet-root', type=str, help='Root directory of the ModelNet dataset.')
#parser.add_argument('--categories', type=str, nargs='+', default=['chair', 'sofa'], help='list of object classes to use.')
parser.add_argument('--num-points', type=int, default=1024, help='Number of points to sample from meshes.')
parser.add_argument('--epochs', type=int, default=10, help='Number of train epochs.')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use.')

args = parser.parse_args()


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
val_loader = DataLoader(val_dataset,batch_size=args.batch_size // 2, shuffle=True)

test_dataset = Scan2CAD(data_frame,split='test',transform=transform, device=args.device)
test_loader = DataLoader(test_dataset,batch_size=1, shuffle=True)

#Same num_classes for all datasets
num_cad_classes = train_dataset.get_num_classes()
model = PointNetClassifier(num_classes=num_cad_classes).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# for e in range(args.epochs):

#     print('###################')
#     print('Epoch:', e)
#     print('###################')

#     train_loss = 0.
#     train_accuracy = 0.
#     num_batches = 0

#     model.train()

#     optimizer.zero_grad()
#     for idx, batch in enumerate(tqdm(train_loader)):
#         pred = model(batch[0])
#         loss = criterion(pred, batch[1].view(-1))
#         train_loss += loss.item()
#         loss.backward()
#         optimizer.step()

#         # Compute accuracy
#         pred_label = torch.argmax(pred, dim=1)
#         train_accuracy += torch.mean((pred_label == batch[1].view(-1)).float()).detach().cpu().item()
#         num_batches += 1

#     print('Train loss:', train_loss / num_batches)
#     print('Train accuracy:', 100 * train_accuracy / num_batches)

#     val_loss = 0.
#     val_accuracy = 0.
#     num_batches = 0

#     model.eval()

#     with torch.no_grad():
#         for idx, batch in enumerate(tqdm(val_loader)):
#             pred = model(batch[0])
#             loss = criterion(pred, batch[1].view(-1))
#             val_loss += loss.item()

#             # Compute accuracy
#             pred_label = torch.argmax(pred, dim=1)
#             val_accuracy += torch.mean((pred_label == batch[1].view(-1)).float()).cpu().item()
#             num_batches += 1

#     print('Val loss:', val_loss / num_batches)
#     print('Val accuracy:', val_accuracy / num_batches)

# test_loader = DataLoader(ModelNet(args.modelnet_root, categories=args.categories,
#                                   split='test', transform=transform, device=args.device),
#                          shuffle=True, batch_size=15)

preds = []
model.eval()

with torch.no_grad():
    for idx, test_batch in enumerate(test_loader):
        pred = model(test_batch)
        pred_labels = torch.max(pred,axis=1)[1]
        print(pred)
        assert 3==2

# test_batch, labels = next(iter(test_loader))
# preds = model(test_batch)
# pred_labels = torch.max(preds, axis=1)[1]

# visualize_batch(test_batch, pred_labels, labels, args.categories)
