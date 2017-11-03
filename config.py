#python 2.7
#tensorflow 1.3.0
data_read_dir = '/Users/yiannilaloudakis/Desktop/basecalling/171016_large/val'
data_write_dir = '/Users/yiannilaloudakis/Desktop/basecalling/171016_large/val_parsed'
data_train_dir = '/Users/yiannilaloudakis/Desktop/basecalling/171016_large/train_parsed'
data_val_dir = '/Users/yiannilaloudakis/Desktop/basecalling/171016_large/val_parsed'
pred_dir = data_val_dir
save_dir = '/Users/yiannilaloudakis/Desktop/basecalling/savedModels'
log_dir =  '/Users/yiannilaloudakis/Desktop/basecalling/logs'
pred_output_dir = '/Users/yiannilaloudakis/Desktop/basecalling/predictions'

model = 'SuperBaseline'
experiment = '1'

max_seq_len = 300
max_base_len = 75

lstm_size = 100
dropout_keep = .8

train = True #We need different helpers for attention decoding!
batch = 32
num_epochs = 1
lr = .001
val_every = 6
restart = True #If false then it restores the model from the ckpt file specified in save_dir