#python 2.7
#tensorflow 1.3.0
data_read_dir = './171016_large/train'
data_write_dir = './171016_large/train_parsed'
data_train_dir = './171016_large/train_parsed'
data_val_dir = './171016_large/val_parsed'
pred_dir = data_val_dir
save_dir = './savedModels'
pred_output_dir = './predictions'

model = 'Achilles'
experiment = '3'

max_seq_len = 300
max_base_len = 75

lstm_size = 100
dropout_keep = .8

train = True #Do you want to train or pred?
batch = 32
max_step = 20
lr = .001
val_every = 5
restart = True #If false then it restores the model from the ckpt file specified in save_dir

normalize = False #Honestly don't know what to do about this one...
verbose = True
print_every = 100