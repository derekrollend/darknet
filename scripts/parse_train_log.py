import sys
import re
import numpy as np
import matplotlib.pyplot as plt

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(argv=None):
    if argv is None:
        argv = sys.argv

    if len(argv) < 2:
        raise Usage("Usage: parse_train_log.py <training_log_file.txt>")

    f = open(argv[1], 'r')

    losses, avg_losses, avg_ious = [], [], []

    for line in f:
        m = re.search('\d+:', line)
        if m:
            parts = line.replace(',', '').split(' ')
            losses.append(float(parts[1]))
            avg_losses.append(float(parts[2]))

        elif line.find('Detection Avg IOU:') >= 0:
            avg_ious.append(float(line.replace(',', '').split(' ')[3]))

    losses_np, avg_losses_np, avg_ious_np = np.asarray(losses), np.asarray(avg_losses), np.asarray(avg_ious)

    print 'IOU Length: ' + str(len(avg_ious_np))
    avg_ious_np = avg_ious_np[0::20]
    print 'After slicing: ' + str(len(avg_ious_np))

    plt.figure()
    plt.plot(losses_np, 'b')
    plt.plot(avg_losses_np, 'r')
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.figure()
    plt.plot(avg_ious_np, 'b')
    plt.title('Average Detection IOUs')
    plt.xlabel('Iteration')
    plt.ylabel('Avg. IOU')
    plt.show()

if __name__ == "__main__":
    sys.exit(main())
