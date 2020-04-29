
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

def get_predictions(args):
    transform = tfs.Compose([
        tfs.TriangleMeshToPointCloud(num_samples=1024),
        tfs.NormalizePointCloud()
    ])

    data_path = args.csv_path
    data_frame = pd.read_csv(data_path)
    data_frame.rename(columns={data_frame.columns[0]:'Filepath', data_frame.columns[1]:'ID'}, inplace=True)

    #Instantiates dataset and dataloader for the data from the csv file
    dataset = Scan2CAD(data_frame, split='full-test', transform=transform, device = args.device)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False)


    # From original data there are 316 classes
    model = PointNetClassifier(num_classes=dataset.num_classes).to(args.device)
    model.load_state_dict(torch.load('pointnet_model_state_dict.pt'))

    model.eval()

    test_predictions = {}

    print("Getting Predictions")

    pred_id_map = pd.read_csv('pred_label_map.csv')
    pred_id_map = pred_id_map.to_dict()
    print(pred_id_map)

    #TODO: ENSURE MAPS ARE CORRECT
    assert 3==2

    #assuming test-batch 1
    with torch.no_grad():
        for idx, filepath, data in enumerate(tqdm(dataloader)):
            pred = model(data)
            pred_labels = torch.argmax(pred, dim=1)
            test_predictions[filepath] = pred_id_map[pred_labels[0].int()]

        
    final_predictions_df = pd.DataFrame.from_dict(test_predictions, orient='index')
    final_predictions_df.rename(columns={data_frame.columns[0]:'Filepath', data_frame.columns[1]:'Predicted ID'}, inplace=True)
    print(final_predictions_df)
    #TODO: ENSURE PREDICTIONS ARE WORKING
    final_predictions_df.to_csv(path_or_buf='predictions.csv')

    print("Testing Completed")
    print("Predictions saved in this directory. Look for predictions.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #Dataset CSV path
    parser.add_argument('--csv_path', type=str, help = 'Path to CSV containing Filepaths - CAD IDs')
    parser.add_argument('--device', type=str, default='available', choices = ['cuda','cpu','available'],
                        help='Device to use (Optional, uses cuda if available!)')

    args = parser.parse_args()
    
    if(args.device == 'available'):
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(args, end="\n\n")

    get_predictions(args)