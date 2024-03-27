
from torch.utils.data import DataLoader
from data.data_processing import MultiFrameDataset, MultiCompFrameDataset, rescale_data, multi_rescale_data, global_rescale
from config import data_path, component, scaling_factor, train_start, train_end, rescale, in_channels, test_start, test_end, sequence_length
from config import mean_value 


import os
from importlib import import_module
import torch
import torch.nn as nn


# create datasets
if in_channels == 1: 
    #train_data, train_scaler = rescale_data(data_path, component, train_start, train_end, scaler=None)
    #test_data, _ = rescale_data(data_path, component, test_start, test_end, scaler=train_scaler)

    train_data, train_scale = global_rescale(data_path, train_start, train_end, custom_scale=None)
    test_data, _ = global_rescale(data_path, test_start, test_end, custom_scale=train_scale) 
    
    train_dataset = MultiFrameDataset(train_data, component, scaling_factor, sequence_length, mean_value, 1.0, train_start, train_end)
    test_dataset = MultiFrameDataset(test_data, component, scaling_factor, sequence_length, mean_value, 1.0, test_start, test_end)

    #train_dataset = MultiFrameDataset(train_data, component, scaling_factor, sequence_length, mean_value, 1.0) 
    #test_dataset = MultiFrameDataset(test_data, component, scaling_factor, sequence_length, mean_value, 1.0)

elif in_channels == 2: 
    train_data, train_scale = global_rescale(data_path, train_start, train_end, custom_scale=None)
    test_data, _ = global_rescale(data_path, test_start, test_end, custom_scale=train_scale) 

    train_dataset = MultiCompFrameDataset(train_data, component[0], component[1], scaling_factor, sequence_length, mean_value, (1.0, 1.0)) 
    test_dataset = MultiCompFrameDataset(test_data, component[0], component[1], scaling_factor, sequence_length, mean_value, (1.0, 1.0))

else: 
    raise ValueError("Unsupported number of in_channels. Expected 1 or 2, got {}".format(in_channels))


class Data:
    def __init__(self, args):
        self.args = args
        self.data_train = args.data_train
        self.data_test = args.data_test

        if not self.args.test_only:

            self.loader_train = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                pin_memory=not self.args.cpu
            )
        else:
            self.loader_train = None

        self.loader_test = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=not self.args.cpu) # modify batch size here




class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        #print('Making model...')
        self.args = args
        self.scale = args.scale
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.ckp = ckp

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.get_model(), file=ckp.log_file)
    
    def forward(self, *args):
        return self.model(*args)
    
    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, is_best=False, filename=''):
        target = self.get_model()
        filename = 'model_{}'.format(filename)
        torch.save(
            target.state_dict(), 
            os.path.join(apath, 'model', '{}latest.pt'.format(filename))
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', '{}best.pt'.format(filename))
            )

    def load(self, apath, pre_train='.', resume=False, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                torch.load(pre_train, **kwargs),
                strict=False
            )

        elif resume:
            print('Loading model from {}'.format(os.path.join(apath, 'model', 'model_latest.pt')))
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        elif self.args.test_only:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_best.pt'),
                    **kwargs
                ),
                strict=False
            )
            

