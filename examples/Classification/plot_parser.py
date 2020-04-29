import os
import matplotlib.pyplot as plt
import numpy as np

train_loss = np.load(os.getcwd() + "/successful_runs_2/train_loss.npy")
val_loss = np.load(os.getcwd() + "/successful_runs_2/val_loss.npy")
print(train_loss)
print(train_loss.shape)

train_acc = np.load(os.getcwd() + "/successful_runs_2/train_acc.npy")
val_acc = np.load(os.getcwd() + "/successful_runs_2/val_acc.npy")

# plt.figure()
# plt.plot(np.array([100*i for i in train_acc]), label='Training Accuracy')
# plt.plot(np.array([100 * i for i in val_acc]), label = 'Validation Accuracy')
# plt.xlabel("Epoch Number")
# plt.ylabel("Accuracy (%)")
# plt.legend()
# plt.title("Training and Validation Accuracies of Pointnet")
# plt.savefig("accuracies.png")

# plt.figure()
# plt.plot(np.array(train_loss), label='Training Loss')
# plt.plot(np.array(val_loss), label = 'Validation Loss')
# plt.xlabel("Epoch Number")
# plt.ylabel("Loss (Cross-Entropy)")
# plt.legend()
# plt.title("Training and Validation Losses of Pointnet")
# plt.savefig("losses.png")