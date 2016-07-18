import os
import numpy as np
import matplotlib.pyplot as plt
from parse import *
import progressbar
import math
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

avg_iter = 40

root_path = "/home/jaehyun/github/torch-inception-resnet-v2"
train_log_filename = "cond1_b64_s12800_i1801710_20160519-193812-101174441.log.train"
val_log_filename = "cond1_b64_s12800_i1801710_20160519-193812-101174441.log.test"

train_log_path = os.path.join(root_path, 'logs', train_log_filename)
val_log_path = os.path.join(root_path, 'logs', val_log_filename)

###################################### train
# Open train_log_path 
with open(train_log_path, 'rt') as f:
    lines = f.readlines()
num_data = len(lines)

# Init necessary variables 
train_axis = np.zeros(num_data)
train_loss = np.zeros(num_data)
train_top1 = np.zeros(num_data)
train_top5 = np.zeros(num_data)
train_axis_avg = np.zeros(math.floor(num_data/avg_iter))
train_loss_avg = np.zeros(math.floor(num_data/avg_iter))
train_top1_avg = np.zeros(math.floor(num_data/avg_iter))
train_top5_avg = np.zeros(math.floor(num_data/avg_iter))

# Init bar and do parsing
print "parse train log" 
bar = progressbar.ProgressBar(maxval=num_data, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for i in xrange(num_data):
  tokens = lines[i].split()

  train_loss[i] = float(tokens[8])
  train_top1[i] = float(tokens[10])
  train_top5[i] = float(tokens[12])

  iters = parse("[{}][{}/{}]", tokens[2])
  train_axis[i] = (float(iters[0])-1) + float(iters[1])/float(iters[2])
  num_iter_per_epoch = float(iters[2])

  if (i+1) % avg_iter == 0:
    j = math.floor((i+1) / avg_iter)-1
    train_axis_avg[j] = train_axis[i]
    train_loss_avg[j] = 0
    train_top1_avg[j] = 0
    train_top5_avg[j] = 0
    for ii in xrange(avg_iter):
      train_loss_avg[j] = train_loss_avg[j] + train_loss[i-ii] / avg_iter
      train_top1_avg[j] = train_top1_avg[j] + train_top1[i-ii] / avg_iter
      train_top5_avg[j] = train_top5_avg[j] + train_top5[i-ii] / avg_iter
     
  bar.update(i+1)
bar.finish()

###################################### val
# Open val_log_path
with open(val_log_path, 'rt') as f:
    lines = f.readlines()
num_data = len(lines)

# Init necessary variables
val_axis = np.zeros(num_data)
val_top1 = np.zeros(num_data)
val_top5 = np.zeros(num_data)

# Init bar and do parsing
print "parse val log" 
bar = progressbar.ProgressBar(maxval=num_data, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for i in xrange(num_data):
  tokens = lines[i].split()

  val_axis[i] = float(tokens[4])
  val_top1[i] = float(tokens[6])
  val_top5[i] = float(tokens[8])

  bar.update(i+1)
bar.finish()

###################################### plot
fig, ax = plt.subplots()
plt.xlim(0, max(train_axis[-1], val_axis[-1]))
plt.ylim(0, 100)
plt.plot(train_axis_avg[::num_iter_per_epoch/avg_iter], train_top1_avg[::num_iter_per_epoch/avg_iter], 'r', label='train (top1)')
plt.plot(train_axis_avg[::num_iter_per_epoch/avg_iter], train_top5_avg[::num_iter_per_epoch/avg_iter], 'r--', label='train (top5)')
plt.plot(val_axis, val_top1, 'b', label='val (top1)')
plt.plot(val_axis, val_top5, 'b--', label='val (top5)')
plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize=11)
plt.grid(True)
plt.yticks(np.arange(0, 101, 10))
ax.yaxis.set_minor_locator(MultipleLocator(5))
plt.xlabel('epochs', fontsize=14, color='black')
plt.ylabel('error (%)', fontsize=14, color='black')
plt.savefig('result')
#plt.show()
