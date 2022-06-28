import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import test_single_volume
from utils import test_single_slide,test_single_slide_bkp

from network.DAUnet_final import DAUnet
from network.SwinUnet import SwinUnet
from network.TransUnet import TransUnet
from network.ConvNeXtUnet import ConvNeXtUnet
from network.Unet import Unet

from trainer import trainer
from config import get_config

from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_AGGC import AGGC_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, help='output dir')   
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=301, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )

# add
parser.add_argument('--network', type=str, default='DAUnet',help='the model of network') 

parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
config = get_config(args)

def inference_AGGC_bkp(args, model, test_save_path=None):
    db_test = args.Dataset( list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):   
        l_image, g_image, label, case_name = sampled_batch["L_image"], sampled_batch["G_image"], sampled_batch["mask"], sampled_batch['case_name']
        # print(l_image.shape, g_image.shape, label.shape)
        metric_i = test_single_slide_bkp(l_image, g_image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing, i_batch=i_batch,network=args.network)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s f1 %f precision %f recall %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d f1 %f precision %f recall %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2]))
    performance = np.mean(metric_list, axis=0)[0]
    precision = np.mean(metric_list, axis=0)[1]
    recall = np.mean(metric_list, axis=0)[2]
    logging.info('Testing performance in best val model: f1 %f precision %f recall %f' % (performance, precision, recall))
    return "Testing Finished!"

def inference_AGGC(args, model, test_save_path=None):
    test_full_image_path = open(os.path.join(args.list_dir, 'test_full_image.txt')).readlines()
    test_full_mask_path = open(os.path.join(args.list_dir, 'test_full_mask.txt')).readlines()

    logging.info("{} test iterations per epoch".format(len(test_full_image_path)))
    model.eval()
    metric_list = 0.0
    for i_batch in tqdm(range(len(test_full_image_path))):   
        image_path, label_path, case_name = test_full_image_path[i_batch].strip('\n'), test_full_mask_path[i_batch].strip('\n'), os.path.basename(test_full_image_path[i_batch]).strip('\n')
        metric_i = test_single_slide(image_path, label_path, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing, i_batch=i_batch,network=args.network)
        print(np.array(metric_i))
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s f1 %f precision %f recall %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2]))
    metric_list = metric_list / len(test_full_image_path)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d f1 %f precision %f recall %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2]))
    performance = np.mean(metric_list, axis=0)[0]
    precision = np.mean(metric_list, axis=0)[1]
    recall = np.mean(metric_list, axis=0)[2]
    logging.info('Testing performance in best val model: f1 %f precision %f recall %f' % (performance, precision, recall))
    return "Testing Finished!"

def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"

if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': { 
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
        'AGGC': { 
            'Dataset': AGGC_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_AGGC',
            'num_classes': 6,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    if args.network == "DAUnet":
        net = DAUnet(config, num_classes=args.num_classes).cuda()
    elif args.network == "SwinUnet":
        net = SwinUnet(config,img_size=args.img_size,num_classes=args.num_classes).cuda()
    elif args.network == "TransUnet":
        net = TransUnet(img_size=args.img_size, num_classes=args.num_classes).cuda()
    elif args.network == "Unet":
        net = Unet(num_classes=args.num_classes).cuda()
    elif args.network == "ConvNeXtUnet":
        net = ConvNeXtUnet(config=config, num_classes=args.num_classes).cuda()
    else:
        raise NotImplementedError("NotImplemented network")
    
    snapshot = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    for k,v in torch.load(snapshot).items():
        print(k)
    
    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained DAUnet",msg)
    snapshot_name = snapshot.split('/')[-1]

    log_folder = args.output_dir + '/test_log_/' + dataset_name
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    # inference(args, net, test_save_path)
    inference_AGGC_bkp(args, net, test_save_path)
