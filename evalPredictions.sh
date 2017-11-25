# Convenience script for analyzing predictions after model has been trained
# NOTE: Do not forget to set the 'train' variable in config to False!

# Get predictions
echo "--Computing predictions"
python trainMaster.py

# Get CSV of predictions
echo "--Writing predictions to CSV"
python hdf5tocsv.py

# Get CSV of gold labels
# NOTE: set the 'data_read_dir' variable in config to val!
echo "--Writing gold labels to CSV"
python getSequences.py

# Analyze predictions
echo "--Analyzing predictions"
python analyzePredictionsLineByLine.py