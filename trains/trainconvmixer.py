import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.dataset import *
from models.convmixer import *
import argparse
from models.utils_gray import *
from metrics.metrics import *
from torch.utils.tensorboard import SummaryWriter

from metrics.loss import *

from utility.metriclogger import *
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = get_logger(LOGER_PATH)
    # Read annotation
    # df_all = pd.read_csv(TRAIN_CSV)

    # train_dataset = CellDataset(TRAIN_PATH, df_all, patch_size=PATCH_SIZE, split='train')
    # 注：python有bug，pickle一次读进大于4G的数据时，在windows上运行会出现EOFError: Ran out of input的错误，解决方案为要么不读取大于4G的数据，要么workers改为0
    train_dataset = KaggleData(is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # val_dataset = CellDataset(TRAIN_PATH, df_all, patch_size=PATCH_SIZE, split='val')
    val_dataset = KaggleData(is_train=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # df_all = pd.read_csv(TRAIN_CSV)
    # train_dataset = CellDataset(TRAIN_PATH, df_all, patch_size=PATCH_SIZE, split='train')
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    #
    # val_dataset = CellDataset(TRAIN_PATH, df_all, patch_size=PATCH_SIZE, split='val')
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    parser = argparse.ArgumentParser(description='Convmixer')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run(default: 400)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--save_freq', type=int, default=10)

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
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--gray', default='no', type=str)

    ##########################################################################
    '''
    python trainconvmixer.py 
    --direc 'path for results to be saved' 
    --epoch 400 --save_freq 10 
    --learning_rate 0.001 
    --gray "no"
    '''
    ##########################################################################
    epochs = 400
    save_freq = 1
    learning_rate = 0.001
    modelname = "Convmixer"

    args = parser.parse_args()
    gray_ = "yes"
    aug = args.aug
    # direc = RESULT_DIR

    if gray_ == "yes":
        from models.utils_gray import JointTransform2D, ImageToImage2D, Image2D
        imgchant = 1
    else:
        from models.utils import JointTransform2D, ImageToImage2D, Image2D
        imgchant = 3

    if args.crop is not None:
        crop = (args.crop, args.crop)
    else:
        crop = None

    # Size of a patch, $p$
    patch_size: int = 8
    # Number of channels in patch embeddings, $h$
    d_model: int = 256
    # Number of [ConvMixer layers](#ConvMixerLayer) or depth, $d$
    n_layers: int = 20
    # Kernel size of the depth-wise convolution, $k$
    kernel_size: int = 7
    # Number of classes in the task
    n_classes: int = 1

    model = ConvMixer(ConvMixerLayer(d_model, kernel_size), n_layers,
                      PatchEmbeddings(d_model, patch_size, 1),
                      ClassificationHead(d_model)).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()), lr=learning_rate,
                                  weight_decay=1e-5)
    # criterion = LogNLLLoss()
    metric = IoUScore()
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()
    best_metric = -1
    best_metric_epoch = -1
    val_loss_values = list()
    epoch_loss_values = list()
    iou_list = list()

    logger.info('start training!')
    for epoch in range(epochs):

        epoch_running_loss = 0
        print("-" * 10)
        print(f"epoch {epoch + 1}/{400}")

        step = 0
        for batch_idx, (X_batch, y_batch) in tqdm(enumerate(train_loader), total =len(train_loader)):
            step += 1
            X_batch = Variable(X_batch.to(device='cuda'))
            y_batch = Variable(y_batch.to(device='cuda'))
            # print(X_batch)
            # print(y_batch)
            # ===================forward=====================

            output = model(X_batch)

            # tmp2 = y_batch.detach().cpu().numpy()
            # tmp = output.detach().cpu().numpy()
            # tmp[tmp >= 0.5] = 1
            # tmp[tmp < 0.5] = 0
            # tmp2[tmp2 > 0] = 1
            # tmp2[tmp2 <= 0] = 0
            # tmp2 = tmp2.astype(int)
            # tmp = tmp.astype(int)

            # yHaT = tmp
            # yval = tmp2

            # 报错，crossentropy需要float point而不是byte，故强转
            # output = output.detach().cpu().numpy()
            # y_batch = y_batch.detach().cpu().numpy()
            # output = torch.FloatTensor(output)
            # y_batch = torch.FloatTensor(y_batch)
            # output = output.float()
            # y_batch = y_batch.float()
            loss = criterion(output, torch.squeeze(y_batch).long())
            # loss = Variable(loss, requires_grad = True)

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_running_loss += loss.item()
            epoch_len = len(train_dataset) // train_loader.batch_size
            # ===================log========================
            writer.add_scalar("train_loss", loss.item())
        epoch_running_loss /= step
        epoch_loss_values.append(epoch_running_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_running_loss:.6f}")

        if epoch == 10:
            for param in model.parameters():
                param.requires_grad = True
        if (epoch % save_freq) == 0:
            model.eval()
            with torch.no_grad():
                for batch_idx, (X_batch, y_batch) in tqdm(enumerate(val_loader), total =len(val_loader)):
                    # print(batch_idx)
                    # if isinstance(rest[0][0], str):
                    #     image_filename = rest[0][0]
                    # else:
                    image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

                    X_batch = Variable(X_batch.to(device='cuda'))
                    y_batch = Variable(y_batch.to(device='cuda'))
                    # start = timeit.default_timer()
                    y_out = model(X_batch)
                    iou_score = metric(y_out, y_batch)
                    val_loss = criterion(y_out, torch.squeeze(y_batch).long())
                    iou_list.append(iou_score)
                    val_loss_values.append(val_loss.detach().cpu().numpy())
                    # stop = timeit.default_timer()
                    # print('Time: ', stop - start)
                    # tmp2 = y_batch.detach().cpu().numpy()
                    # tmp = y_out.detach().cpu().numpy()
                    # tmp[tmp >= 0.5] = 1
                    # tmp[tmp < 0.5] = 0
                    # tmp2[tmp2 > 0] = 1
                    # tmp2[tmp2 <= 0] = 0
                    # tmp2 = tmp2.astype(int)
                    # tmp = tmp.astype(int)

                    # print(np.unique(tmp2))
                    # yHaT = tmp
                    # yval = tmp2

                    epsilon = 1e-20

                    del X_batch, y_batch, y_out

                    # yHaT[yHaT == 1] = 255
                    # yval[yval == 1] = 255
                    # fulldir = direc + "/{}/".format(epoch)
                    # print(fulldir+image_filename)
                    # if not os.path.isdir(fulldir):
                    # os.makedirs(fulldir)

                    # cv2.imwrite(fulldir + image_filename, yHaT[0, 1, :, :])
                avg_iou = np.mean(iou_list)
                avg_val_loss = np.mean(val_loss_values)
                if avg_iou > best_metric:
                    best_metric = avg_iou
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_convmixermodel_segmentation_array.pth")
                    print("saved new best metric model")
                print("current epoch: {} current mean val loss: {:.6f} current mean iou: {:.6f} best mean iou: {:.6f} at epoch {}".format(epoch + 1, avg_val_loss, avg_iou, best_metric, best_metric_epoch))
                writer.add_scalar("val_mean_iou", avg_iou, epoch + 1)
                logger.info('Epoch:[{}/{}]\t loss={:.6f}\t avg_val_loss={:.6f}\t avg_iou={:.6f}'.format(epoch, epochs, epoch_running_loss, avg_val_loss, avg_iou))
                model.train()

    logger.info('finish training!')
    writer.close()



if __name__ == '__main__':
    main()
