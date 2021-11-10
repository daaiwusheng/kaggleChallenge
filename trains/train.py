
import pandas as pd

import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2

from models.utils_gray import *
from models.medtnet import *
from models.dataset import *
from metrics.metrics import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read annotation
df_all = pd.read_csv(TRAIN_CSV)


train_dataset = CellDataset(TRAIN_PATH, df_all, patch_size=PATCH_SIZE, split='train')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

val_dataset = CellDataset(TRAIN_PATH, df_all, patch_size=PATCH_SIZE, split='val')
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--train_dataset', required=False, type=str)
parser.add_argument('--val_dataset', type=str)
parser.add_argument('--save_freq', type=int, default=10)

parser.add_argument('--modelname', default='MedT', type=str,
                    help='type of model')
parser.add_argument('--cuda', default="on", type=str,
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str,
                    help='load a pretrained model')
parser.add_argument('--save', default='default', type=str,
                    help='save the model')
parser.add_argument('--direc', default='./medt', type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--imgsize', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--gray', default='no', type=str)

##########################################################################
'''
python train.py 
--train_dataset "enter train directory" 
--val_dataset "enter validation directory" 
--direc 'path for results to be saved' 
--batch_size 4 
--epoch 400 --save_freq 10 
--modelname "gatedaxialunet" 
--learning_rate 0.001 
--imgsize 128 
--gray "no"
'''
##########################################################################

args = parser.parse_args()
gray_ = "yes"
aug = args.aug
args.batch_size = 32
direc = "/databig/kaggle_result/"
modelname = "convmixer"
imgsize = 64


if gray_ == "yes":
    import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 1
else:
    from utils import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 3

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None



model = logo(img_size = imgsize, imgchan = imgchant)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model, device_ids=[0, 1]).cuda()

model.to(device)


optimizer = torch.optim.AdamW(list(model.parameters()), lr=args.learning_rate,
                             weight_decay=1e-5)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.set_deterministic(True)
# random.seed(seed)


for epoch in range(args.epochs):

    epoch_running_loss = 0

    for batch_idx, (X_batch, y_batch, *rest) in enumerate(train_loader):
        X_batch = Variable(X_batch.to(device='cuda'))
        y_batch = Variable(y_batch.to(device='cuda'))

        # ===================forward=====================

        output = model(X_batch)

        tmp2 = y_batch.detach().cpu().numpy()
        tmp = output.detach().cpu().numpy()
        tmp[tmp >= 0.5] = 1
        tmp[tmp < 0.5] = 0
        tmp2[tmp2 > 0] = 1
        tmp2[tmp2 <= 0] = 0
        tmp2 = tmp2.astype(int)
        tmp = tmp.astype(int)

        yHaT = tmp
        yval = tmp2

        loss = iou_numpy(output, y_batch)

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_running_loss += loss.item()

    # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch, args.epochs, epoch_running_loss / (batch_idx + 1)))

    if epoch == 10:
        for param in model.parameters():
            param.requires_grad = True
    if (epoch % args.save_freq) == 0:

        for batch_idx, (X_batch, y_batch, *rest) in enumerate(val_loader):
            # print(batch_idx)
            if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
            else:
                image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

            X_batch = Variable(X_batch.to(device='cuda'))
            y_batch = Variable(y_batch.to(device='cuda'))
            # start = timeit.default_timer()
            y_out = model(X_batch)
            # stop = timeit.default_timer()
            # print('Time: ', stop - start)
            tmp2 = y_batch.detach().cpu().numpy()
            tmp = y_out.detach().cpu().numpy()
            tmp[tmp >= 0.5] = 1
            tmp[tmp < 0.5] = 0
            tmp2[tmp2 > 0] = 1
            tmp2[tmp2 <= 0] = 0
            tmp2 = tmp2.astype(int)
            tmp = tmp.astype(int)

            # print(np.unique(tmp2))
            yHaT = tmp
            yval = tmp2

            epsilon = 1e-20

            del X_batch, y_batch, tmp, tmp2, y_out

            yHaT[yHaT == 1] = 255
            yval[yval == 1] = 255
            fulldir = direc + "/{}/".format(epoch)
            # print(fulldir+image_filename)
            if not os.path.isdir(fulldir):
                os.makedirs(fulldir)

            cv2.imwrite(fulldir + image_filename, yHaT[0, 1, :, :])
        fulldir = direc + "/{}/".format(epoch)
        torch.save(model.state_dict(), fulldir + args.modelname + ".pth")
        torch.save(model.state_dict(), direc + "final_model.pth")

