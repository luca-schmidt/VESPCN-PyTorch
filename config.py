import os
data_path = "/mnt/qb/work/ludwig/lqb424/datasets/wind_data.nc"
#component = ("u10", "v10")
component = "u10"
scaling_factor = 4
train_start = "2017-01-01"
train_end = "2019-12-31" 
test_start = "2020-01-01"
test_end = "2020-12-31"
rescale = True
in_channels = 1
comp_idx = 0
#mean_value = (0.5027, 0.5396)
mean_value = 0.5027
sequence_length = 3
#model_path = "src/models/mf_espcn"
#model_name = f'mf_espcn_{scaling_factor}x'
##output_dir = f'src/models/mf_espcn/results_{model_name}'
#config_dir = os.path.join(output_dir, 'config.json')
#model_dir = f'{output_dir}/pytorch_model_{scaling_factor}_{component}x.pt'


