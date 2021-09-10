"""Params for ADDA."""

num_classes = 16
use_coral = True

# params for dataset and data loader
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 16
image_size = 64

# params for source dataset
# src_dataset = "MNIST"
src_dataset = "16_class_source"
src_encoder_restore = "snapshots/ADDA-source-encoder-final.pt"
src_classifier_restore = "snapshots/ADDA-source-classifier-final.pt"
src_model_trained = True

# params for target dataset
tgt_dataset = "16_class_target"
tgt_encoder_restore = "snapshots/ADDA-target-encoder-final.pt"
tgt_model_trained = True

# params for setting up models
model_root = "snapshots"
# d_input_dims = 500
d_input_dims = 2048
d_hidden_dims = 500
d_output_dims = 2
d_model_restore = "snapshots/ADDA-critic-final.pt"

# params for training network
num_gpu = 1
num_epochs_pre = 20
log_step_pre = 20
eval_step_pre = 5
save_step_pre = 10
num_epochs = 2000
log_step = 100
save_step = 100
manual_seed = None

# ONLY FOR DEBUG:
# num_epochs_pre = 5  # 100
# log_step_pre = 1  # 20
# eval_step_pre = 5  # 20
# save_step_pre = 5  # 100
# num_epochs = 10  # 2000
# log_step = 1  # 100
# save_step = 5  # 100

# params for optimizing models
e_learning_rate = 1e-5
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9
