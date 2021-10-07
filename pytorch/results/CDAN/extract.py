import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import matplotlib.pyplot as plt


source_acc_uda = np.load("source_test_acc.npy")
target_acc_uda = np.load("target_test_acc.npy")

source_acc_uda = np.insert(source_acc_uda,0,0.12)
target_acc_uda = np.insert(target_acc_uda,0,0.1)

source_acc_no_uda = np.load("/home/sachingo/CDAN/pytorch_no_uda/source_test_acc_no_uda.npy")
target_acc_no_uda = np.load("/home/sachingo/CDAN/pytorch_no_uda/target_test_acc_no_uda.npy")

source_acc_no_uda = np.insert(source_acc_no_uda,0,0.12)
target_acc_no_uda = np.insert(target_acc_no_uda,0,0.1)

shorter = len(source_acc_no_uda)
source_acc_uda = source_acc_uda[:shorter]
target_acc_uda = target_acc_uda[:shorter]

print("maximum target test acc : ", max(target_acc_uda))

x = np.arange(1, shorter + 1)
# plotting
plt.title("test acc with Condition Adversarial Domain Adaptation")
plt.xlabel("iteration")
plt.ylabel("accuracy")
plt.plot(x, source_acc_uda, label="CIFAR 10.0 Condition Discriminator")
plt.plot(x, target_acc_uda, label="CIFAR 10.1 Condition Discriminator")
plt.plot(x, source_acc_no_uda, label="CIFAR_10.0 standard train")
plt.plot(x, target_acc_no_uda, label="CIFAR_10.1 standard train")
# plt.plot(x, source_acc_rot_onlysu, label="CIFAR_10.0_rot_source_only")
# plt.plot(x, test_acc_rot_onlysu, label="CIFAR_10.1_rot_source_only")
plt.legend(loc="lower right")
# plt.show()
plt.savefig("CDAN.png")
# exit()
