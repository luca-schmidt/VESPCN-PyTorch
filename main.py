import torch
import data
import model
from option import args
from trainer_vsr import Trainer_VSR
from logger import logger
from config import in_channels, comp_idx, scaling_factor, component

print('Number of input channels: {}'.format(in_channels))
print('Evaluate performance on component: {}'.format(comp_idx +1))


args.test_only = False #True: use pre-trained model; False: train from scratch
#args.pre_train = f'experiment/save_path/model/model_{component}_{scaling_factor}x_latest.pt' # if want to use pre-trained model
args.pre_train = '.' # if want to train from scratch


torch.manual_seed(args.seed)
chkp = logger.Logger(args)


if args.task == 'Video':
    print("Selected task: Video")
    model = model.Model(args, chkp)
    loader = data.Data(args)
    t = Trainer_VSR(args, loader, model, chkp)

    if args.test_only:
        print("Testing using pre-trained model.")
        t.test()

    else:
        while not t.terminate():
            print("Training model from scratch.")
            t.train()
            t.test()
            #t.plot_images(num_images=1)

else: 
    print("Task not supported.")


chkp.done()
