# Chiron
## A basecaller for Oxford Nanopore Technologies' sequencers
Using a deep learning CNN+RNN+CTC structure to establish end-to-end basecalling for the nanopore sequencer.
Built with **TensorFlow** and python 2.7.

If you found Chiron useful, please consider to cite:
> Teng, H., et al. (2017). Chiron: Translating nanopore raw signal directly into nucleotide sequence using deep learning. [bioRxiv 179531] (https://www.biorxiv.org/content/early/2017/09/12/179531)

---

## Install


### Installing dependencies

Chiron requires the following packages:
* `Tensorflow >= 1.0.1`
* `numpy >= 1.13.1`
* `h5py >= 2.6`

Follow [this guide](https://www.tensorflow.org/install/) for installing Tensorflow, if possible with GPU support.
You can use `pip` for installing `numpy` and `h5py`:

```
pip install numpy
pip install h5py
```


### Installing Chiron

To install Chiron, first clone the repository to your local machine:

```
git clone https://github.com/cangermueller/Chiron.git
```

To install the Chiron package for code development, execute the following command in the Chiron root directory:

```
python setup.py develop
```

In this way, changes to the source code are visible right away.


## Testing Chiron
To test the installation and train Chiron on a small dataset, decompress the example data and execute the example script:

```
cd ./examples
tar xf ./data.tar.gz
cd ./train
./train.sh
```

## Using chiron
Chiron provides a `train`, `call`, and `export` command for training, base calling, and exporting signal/label files from fast5 files.

```
chiron {train,call,export} [FLAGS]
```

You can get more information about the different command by using the `--help` flag, e.g. `chiron train --help`, or reading the [official Chiron guide](https://github.com/haotianteng/Chiron).


## Contact
* Christof Angermueller
* cangermueller@gmail.com
* https://cangermueller.com
* [@cangermueller](https://twitter.com/cangermueller)
