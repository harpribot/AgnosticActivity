# AgnosticActivity

The project involves separating the object recognition task from activity recognition, so that, the activity recognition is agnostic to the objects that are involves, and thus improves learning when the dataset is heavily skewed towards certain objects performing certain activities, which wont lead to better generalization if different objects are performing the same activity.

# Instructions
## Building Caffe-sl
Follow the following steps:
* Add your makefile.config in caffe-sl directory
* Run 
```
make all -j10
```
* All done

Thats it. Note that this caffe-sl version is stripped off of unwanted files in the original caffe-sl version for the project

## Models
All the models are in the models directory.

## Experiments
All the experiment scripts should be in Experiments directory and should call the models directory from there.

## Utilities
There are some utility scripts that we can use. It is not properly strucutred. As the project is in progress


