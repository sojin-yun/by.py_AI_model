
import numpy as np
import matplotlib.pyplot as plt

x1 = range(15)
#x1 = range(len(train_acc))
#x2 = range(len(valid_acc))

train_loss = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,80]
valid_loss = [5,4,3,2,1,5,4,3,2,1,5,4,3,2,1]
train_acc = [4,5,6,7,8,4,5,6,7,8,4,5,6,7,8]
valid_acc = [5,7,8,2,14,5,7,8,2,14,5,7,8,2,14]

plt.subplot(1,2,1), plt.grid(), plt.tick_params(labelsize=8)
plt.plot(x1, train_loss, 'b', label = 'Training loss')
plt.plot(x1, valid_loss, 'r', label = 'Validation loss')
plt.xlabel('epoch'), plt.title('Loss', size=10)
plt.legend(loc='upper right', fontsize=8)

plt.subplot(1,2,2), plt.grid(), plt.tick_params(labelsize=8)
plt.plot(x1, train_acc, 'b', label = 'Training accuracy')
plt.plot(x1, valid_acc, 'r', label = 'Validation accuracy')
plt.xlabel('epoch'), plt.title('Accuracy', size=10)
plt.legend(loc='upper right', fontsize=8)

plt.suptitle("Main Title")
plt.tight_layout()
plt.savefig('img.png')