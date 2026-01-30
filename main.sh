ml python/3.12.3
mkdir results
mkdir logs
mkdir datasets

#Setup3 and scalability are commented out since
#they require manually installed datasets. If
#you have installed Twitter and Uber, you may run them.
pip install -r requirements.txt
python3 src/createcsv.py
./runExperimentalSetup1.sh
./runExperimentalSetup2.sh
#./runExperimentalSetup3.sh
#./runExperimentalScalability.sh
