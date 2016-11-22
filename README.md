# AgnosticActivity

The project involves separating the object recognition task from activity recognition, so that, the activity recognition is agnostic to the objects that are involves, and thus improves learning when the dataset is heavily skewed towards certain objects performing certain activities, which wont lead to better generalization if different objects are performing the same activity.

# Motivation
**Intra Activity Skewness**
![Intra-im](/Experiments/Exp1-VizualizeSkewness/vizualisation/intra-activity.png)
![Intra-plot](/Experiments/Exp1-VizualizeSkewness/vizualisation/intra-plot.png)

**Inter Activity Skewness**
![Inter-implot](/Experiments/Exp1-VizualizeSkewness/vizualisation/inter-activity.png)

# Models
![Models](/models/models.png "Top: Original Model, Bottom 2: Proposed Models")

All the models are in the models directory

# Requirements
## Building Caffe-sl
Follow the following steps:
* Add your makefile.config in caffe-sl directory
* Run `make all -j10`
* All done

Thats it. Note that this caffe-sl version is stripped off of unwanted files in the original caffe-sl version for the project

## Torch and dependencies
* Install [Torch](http://torch.ch/docs/getting-started.html)
* Install [PyTorch](https://github.com/hughperkins/pytorch)
* Requires torch packages: `luarocks install nnlr cudnn rnn` to install.
* Requires python packages: `pip install h5py numpy scikit-learn` to install

## Project data
All the data is to be kept in the project-data directory and queried from there.

## Experiments
All the experiment scripts should be in Experiments directory and should call the models directory from there.

## Utilities
There are some utility scripts that we can use. It is not properly strucutred. As the project is in progress
