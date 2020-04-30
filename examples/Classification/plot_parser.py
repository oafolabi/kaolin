import os
import matplotlib.pyplot as plt
import numpy as np

train_loss = np.load(os.getcwd() + "/data/train_loss_full_merge2_128_50_1.npy")
val_loss = np.load(os.getcwd() + "/data/val_loss_full_merge2_128_50_1.npy")
print(train_loss)
print(train_loss.shape)

train_acc = np.load(os.getcwd() + "/data/train_acc_full_merge2_128_50_1.npy")
val_acc = np.load(os.getcwd() + "/data/val_acc_full_merge2_128_50_1.npy")
print(val_acc)

plt.figure()
plt.plot(np.array([100*i for i in train_acc]), label='Training Accuracy')
plt.plot(np.array([100 * i for i in val_acc]), label = 'Validation Accuracy')
plt.xlabel("Epoch Number")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Training and Validation Accuracies of Pointnet")
plt.savefig("accuracies_merge2_128.png")

plt.figure()
plt.plot(np.array(train_loss), label='Training Loss')
plt.plot(np.array(val_loss), label = 'Validation Loss')
plt.xlabel("Epoch Number")
plt.ylabel("Loss (Cross-Entropy)")
plt.legend()
plt.title("Training and Validation Losses of Pointnet")
plt.savefig("losses_merge2_128.png")