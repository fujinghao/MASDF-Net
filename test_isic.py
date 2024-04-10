import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.MASDF_Net import MASDF_Net
from utils.dataloader import test_dataset
import imageio
def mean_sensitivity_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection
    mask_GT=np.sum(np.abs(y_true), axis=axes)
    smooth = .001
    se = (intersection + smooth) / (mask_GT + smooth)
    return se

def get_specificity(GT, SR, threshold=0.5):
    SR = torch.from_numpy(SR)
    GT = torch.from_numpy(GT)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0) & (GT == 0))
    FP = ((SR == 1) & (GT == 0))

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP
def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1) 
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum  - intersection 
    
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    
    smooth = .001
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    return dice


if __name__ == '__main__':
    dataset = '2018' + '_'
    cross_dataset = '2018' + '_'
    network = 'MASDF_Net'
    parser = argparse.ArgumentParser()
    model = MASDF_Net().cuda()
    parser.add_argument('--ckpt_path', type=str, default='snapshots/' + dataset + network + '/' + 'MASDF_Net.pth')





    parser.add_argument('--test_path', type=str,
                        default='data/' + cross_dataset + '224_224/', help='path to test dataset')
    parser.add_argument('--save_path', type=str, default='save/network_pred/' + cross_dataset + network + '_pred', help='path to save inference segmentation')

    opt = parser.parse_args()

    model.load_state_dict(torch.load(opt.ckpt_path))
    model.cuda()
    model.eval()

    print('evaluating model: ', opt.ckpt_path)

    image_root = '{}/data_test.npy'.format(opt.test_path)
    gt_root = '{}/mask_test.npy'.format(opt.test_path)
    test_loader = test_dataset(image_root, gt_root)

    se_bank = []
    sp_bank = []
    dice_bank = []
    iou_bank = []
    acc_bank = []
    if opt.save_path is not None:
        os.makedirs(opt.save_path, exist_ok=True)

    for i in range(test_loader.size):
        image, gt = test_loader.load_data()
        gt = 1*(gt>0.5)
        image = image.cuda()
        with torch.no_grad():
            res = model(image)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 1*(res > 0.5)

        if opt.save_path is not None:
            imageio.imwrite(opt.save_path+'/'+str(i)+'_pred.jpg', res)

            imageio.imwrite(opt.save_path+'/'+str(i)+'_gt.jpg', gt)
        se = mean_sensitivity_np(gt, res)
        sp = get_specificity(gt, res)
        dice = mean_dice_np(gt, res)
        iou = mean_iou_np(gt, res)
        acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])
        sp_bank.append(sp)
        se_bank.append(se)
        acc_bank.append(acc)
        dice_bank.append(dice)
        iou_bank.append(iou)

    print('Dice: {:.4f}, IoU: {:.4f}, SE: {:.4f}, SP: {:.4f}, Acc: {:.4f}'
          .format(np.mean(dice_bank), np.mean(iou_bank), np.mean(se_bank), np.mean(sp_bank),
                 np.mean(acc_bank)))
