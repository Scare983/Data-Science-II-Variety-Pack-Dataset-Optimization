Daksha Devasthale, Chinmay Deosthali, Kevin Linnane
Data Monks 

Data folder should be one above the Python directory.  

The neural nets uses Keras with tensor flow backend so you need to install both of these.

pip install keras
pip install tensorflow
_________________________________________________________________________
To run and get same Allresults found in each Folder (Aqua, Concrete, Boston, Mpg ):

Run the command:

sh RunTests.sh

or

./RunTests.sh
__________________________________________________________________________

These results are the mass of all variations of activation functions and optimizers using .1 and .2 eta with 200 epochs.
The results also output a graph in the AllResults/Images folder  where x is number of paramters, y is rValue FOR EACH VARIATION

If you only want one eta for a certain file, you can execute:
# boston = fileName without .csv
# .1 = eta value to use
# 200 = # of epochs to use
python General.py XL boston .1 200 > aFile.txt
 
__________________________________________________________________________
To run and get epoch Graphs with best params run:  

sh RunOptimalModels.sh

or 

./RunTests.sh RunOptimalModels.sh
__________________________________________________________________________


After getting all results, we look at the models that performed best and then execute each models bestPerformance.py to get Image results of the epochs
These results are put in the Image folder each Datasetfolder/bestModel/Images.
You can run each individual  Bestfile for their output, ei:

python BestConcrete.py 

__________________________________________________________________________

The Rsq, rCv, and rSq graphs were manually moved from allResults folder after execution once we determine the best model's parameters.  (Not the best way, but quicker than fixing this way and running everything)

