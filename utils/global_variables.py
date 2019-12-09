import os
if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False

UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
