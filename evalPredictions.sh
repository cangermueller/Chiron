# Convenience script for analyzing predictions after model has been trained
# NOTE: Do not forget to set the 'train' variable in config to False!

# Get predictions
echo "--Computing predictions"
python trainMaster.py

# Get CSV of predictions
echo "--Writing predictions to CSV"
python hdf5tocsv.py

# Get CSV of gold labels if necessary
# NOTE: set the 'data_read_dir' variable in config to val!
gold_labels_file="val_labels.csv"
if ! [ -e "$gold_labels_file" ]; then
	echo "--Writing gold labels to CSV"
	python getSequences.py
fi

# Analyze predictions
echo "--Analyzing predictions"
python analyzePredictionsLineByLine.py