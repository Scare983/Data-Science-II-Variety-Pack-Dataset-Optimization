
# take note that the names it is being outputted to are related to inputs.  Python will create figures to put into AllResults/Images with the same inputs you are giving.



echo "# .1 eta Boston"

python  General.py perceptron boston .1 200  >  Boston/Allresults/BostonPerceptron200Epoch1Eta.txt 2>>error.log

## Stuff I need to run ##
python  General.py 3L boston .1 200>  Aqua/Allresults/BostonNN3L200Epoch1Eta.txt 2>>error.log
python  General.py XL boston .1 200 > Aqua/Allresults/BostonNN4L200Epoch1Eta.txt 2>>error.log

echo "# .2 eta Boston"
python  General.py perceptron boston  .2 200  >  Boston/Allresults/BostonPerceptron200Epoch2Eta.txt 2>>error.log
python  General.py 3L boston  .2 200 >  Boston/Allresults/BostonNN3L200Epoch2Eta.txt 2>>error.log
python  General.py XL boston  .2 200 > Boston/Allresults/BostonNN4L200Epoch2Eta.txt 2>>error.log


echo "# .1 eta Aqua"
python  General.py perceptron qsar_aquatic_toxicity .1 200  >  Aqua/Allresults/AquaPerceptron200Epoch1Eta.txt 2>>error.log
python  General.py 3L qsar_aquatic_toxicity .1 200>  Aqua/Allresults/AquaNN3L200Epoch1Eta.txt 2>>error.log
python  General.py XL qsar_aquatic_toxicity .1 200 > Aqua/Allresults/AquaNN4L200Epoch1Eta.txt 2>>error.log

echo "# .2 eta Aqua"
python  General.py perceptron qsar_aquatic_toxicity  .2 200  >  Aqua/Allresults/AquaPerceptron200Epoch2Eta.txt 2>>error.log
python  General.py 3L qsar_aquatic_toxicity  .2 200 >  Aqua/Allresults/AquaNN3L200Epoch2Eta.txt 2>>error.log
python  General.py XL qsar_aquatic_toxicity  .2 200 > Aqua/Allresults/AquaNN4L200Epoch2Eta.txt 2>>error.log

echo "# .1 eta Mpg"
python  General.py perceptron auto-mpg .1 200  >  Mpg/Allresults/MpgPerceptron200Epoch1Eta.txt 2>>error.log
python  General.py 3L auto-mpg .1 200>  Mpg/Allresults/MpgNN3L200Epoch1Eta.txt 2>>error.log
python  General.py XL auto-mpg .1 200 > Mpg/Allresults/MpgNN4L200Epoch1Eta.txt 2>>error.log

echo "# .2 eta Mpg"
python  General.py perceptron auto-mpg  .2 200  >  Mpg/Allresults/MpgPerceptron200Epoch2Eta.txt 2>>error.log
python  General.py 3L auto-mpg  .2 200 >  Mpg/Allresults/MpgNN3L200Epoch2Eta.txt 2>>error.log
python  General.py XL auto-mpg  .2 200 > Mpg/Allresults/MpgNN4L200Epoch2Eta.txt 2>>error.log



# .1 eta Concrete

# Concrete is different because it was a test dummy but too late to fix code. , but it runs each Method and outputs the file
python Concrete.py .1 200 > Concrete/Concrete/AllResults/ 2>>error.log
python Concrete.py .2 200 > Concrete/Concrete/AllResults/ 2>>error.log

echo "# .1 eta Fifa"
python  General.py perceptron fifa_player_data .1 200  >  Aqua/Allresults/FifaPerceptron200Epoch1Eta.txt 2>>error.log
python  General.py 3L fifa_player_data .1 200>  Fifa/Allresults/FifaNN3L200Epoch1Eta.txt 2>>error.log
python  General.py XL fifa_player_data .1 200 > Fifa/Allresults/FifaNN4L200Epoch1Eta.txt 2>>error.log

echo "# .2 eta Fifa"
python  General.py perceptron fifa_player_data  .2 200  >  Fifa/Allresults/FifaPerceptron200Epoch2Eta.txt 2>>error.log
python  General.py 3L fifa_player_data  .2 200 >  Fifa/Allresults/FifaNN3L200Epoch2Eta.txt 2>>error.log
python  General.py XL fifa_player_data  .2 200 > Fifa/Allresults/FifaNN4L200Epoch2Eta.txt 2>>error.log