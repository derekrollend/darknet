import re
import numpy as np
import matplotlib.pyplot as plt

f = open('tiny_train_scratch_11.txt', 'r')

losses, avg_losses = [], []

for line in f:
    m = re.search('\d+:', line)
    if m:
        parts = line.replace(',', '').split(' ')
        losses.append(float(parts[1]))
        avg_losses.append(float(parts[2]))

losses_np, avg_losses_np = np.asarray(losses), np.asarray(avg_losses)

plt.figure()
plt.plot(losses_np, 'b')
plt.plot(avg_losses_np, 'r')
plt.title('Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
