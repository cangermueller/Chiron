#python 2.7
#tensorflow 1.3.0

# Old stuff
# data_write_dir = './171016_large/train_parsed'
# pred_dir = data_val_dir

model = 'Chiron_3'
experiment = '1'

data_train_dir = './171016_large/train_parsed'
data_val_dir = './171016_large/val_parsed'

save_dir = './savedModels'
pred_output_dir = './predictions'

data_read_dir = './171016_large/val'
write_database = './val.hdf5'

train_database = './train.hdf5'
val_database = './val.hdf5'
test_database = './val.hdf5'
predictions_database = pred_output_dir + '/' + model + experiment + 'predictions.hdf5'
test_label_file = 'val_labels.csv'

max_seq_len = 300
max_base_len = 75

lstm_size = 200
dropout_keep = .8

train = True #Do you want to train or pred?
batch = 200
max_step = 30000
lr = .001
val_every = 5
restart = True #If false then it restores the model from the ckpt file specified in save_dir

normalize = True #Honestly don't know what to do about this one...
verbose = True
print_every = 100
