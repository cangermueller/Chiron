#python 2.7
#tensorflow 1.3.0
data_read_dir = '/Users/yiannilaloudakis/Desktop/basecalling/171016_large/val'
data_write_dir = '/Users/yiannilaloudakis/Desktop/basecalling/171016_large/val_parsed'
data_train_dir = '/Users/yiannilaloudakis/Desktop/basecalling/171016_large/train_parsed'
data_val_dir = '/Users/yiannilaloudakis/Desktop/basecalling/171016_large/val_parsed'
save_dir = '/Users/yiannilaloudakis/Desktop/basecalling/savedModels'
max_seq_len = 300
max_base_len = 75
model = 'Baseline'
train = True
batch = 32
num_epochs = 2
lr = .001
val_every = 100
experiment = '1'
restart = False #If false then it restores the model from the ckpt file specified in save_dir