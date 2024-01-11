from pathlib import Path

ROOT_PATH = Path("ROOT_PATH")
DATA_PATH = ROOT_PATH / 'DATA_PATH'
S1_PATH = DATA_PATH / 'S1'
S2_PATH = DATA_PATH / "S2"
LABEL10_PATH = DATA_PATH / "LABEL_10"
WEIGHT_PATH = ROOT_PATH/'WEIGHT_PATH'

IMG_HEIGHT, IMG_WIDTH = 128, 128
s1_ch, s2_ch = 3, 5
model_patch = 128
# dropout_rate = 0.2
splits = 0.2
train_batchSize = 4
val_batchSize = 1
S2_MAX = 3000
lr = 0.0001
maxDepthVal = 176.0

LABEL_fname = 'Filter_LABELS.csv'

ssim_loss_weight = 0.4 #0.85
mse_loss_weight = 0.6
l1_loss_weight = 0.1
edge_loss_weight = 0.9
