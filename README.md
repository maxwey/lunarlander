# lunarlander
Land a lunar!


Dependencies for this project are (pip3 install should do the trick)
- tensorflow
- tflearn
- box2d
- box2d-kengz
- gym
- numpy


There are several files in this repo of note: 
1. `dataGenerator.py`: this file is used to generate data for many thousands of run trials. It has been written to run several processes together to divide up the task. Each process will run however many trials are defined in the variables section of the file. Each process will then save the results of the trials that pass the minimum score threshold to a file. At the end of the run, the files are then read and concatenated together to create a single file with all of the data. (I've noticed a slight bug with this; reading the files using numpy.load one by one and concatenating the arrays together seemed to do the trick). You must create a `data` directory before running this program.

2. `demo` folder contains the file `driverLander.py` and several models. The driver can be run with the name of the model as a command line argument. (Omit the extensions beyond `.tflearn`; for instance to run final_model, use `final_model.tflearn` as the arugment). No arguments will run the default "model-less" version (random moves).

3. `lunarLander.py`: this file is used to create and train the neural network used to select the next best move for the lander. The model is iteratively trained on previous models and uses a data set of the best moves selected from the last generation's episodes. This data set is loaded from file in the function `iterative()`, this can be edited to be a custom file created from `dataGenerator.py`. For each new generation, the current model is saved to a `.tflearn` file. Note: `iterative()` will run infinitely, terminate the program when the model for the current generation is satisfactory.
