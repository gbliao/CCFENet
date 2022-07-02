import argparse
import os
from data_Siam import get_loader
from solver import Solver
import time
import torch
torch.set_num_threads(4)



def get_test_info(config):
    if config.sal_mode == 'VT821':
        test_root = '/userhome/Data/640/VT821/'
    elif config.sal_mode == 'VT1000':
        test_root = '/userhome/Data/640/VT1000/'
    elif config.sal_mode == 'VT5000':
        test_root = '/userhome/Data/640/VT5000/'
    
    else:
        raise Exception('Invalid config.sal_mode')

    config.test_root = test_root           
    

def main(config):
    if config.mode == 'test':
        get_test_info(config)
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.test_folder): 
            os.makedirs(config.test_folder)
        test = Solver(test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    resnet_path = 'pretrained/resnet34.pth'

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--lr', type=float, default=5e-5)  
    parser.add_argument('--wd', type=float, default=0.0005)  
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--image_size', type=int, default=256)              
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--device_id', type=str, default='cuda:0')
    parser.add_argument('--arch', type=str, default='resnet')  
    parser.add_argument('--pretrained_model', type=str, default=resnet_path)  
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)  
    parser.add_argument('--num_thread', type=int, default=4)
    parser.add_argument('--load', type=str, default='')  
    parser.add_argument('--save_folder', type=str, default='saved_model/')       
    parser.add_argument('--epoch_save', type=int, default=5)


    # settings
    parser.add_argument('--model', type=str, default='saved_model/CCFENet.pth')        
    parser.add_argument('--test_folder', type=str, default='Results/VT5000/')                 
    parser.add_argument('--sal_mode', type=str, default='VT5000',
                        choices=['VT821', 'VT1000', 'VT5000', 
                                ])  # Test image dataset

    parser.add_argument('--mode', type=str, default='test', choices=['test'])     
    config = parser.parse_args()

    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)

    get_test_info(config)
    main(config)
