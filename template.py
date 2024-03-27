def set_template(args):
  if  args.template == 'CustomTemplate':
    args.lr = 1e-4
    #args.model = 'ESPCN_multiframe2'
    args.task = 'Video'
    args.data_train = 'VSRData'
    args.data_test = 'VSRData'
    args.process = True
    args.n_sequence = 3
    args.scale = 4
    args.epochs = 100
    args.n_colors = 1 # = number of in_channels
    args.batch_size = 32


 

