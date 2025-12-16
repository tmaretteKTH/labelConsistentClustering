# Label-consistent clustering for evolving data

This code implements algorithms described in "Label-consistent clustering for evolving data", by Ameet Gadekar, Aristides Gionis, and Thibault Marette.

___
## Table of Contents

1. [Reproduce results](#1-reproduce-results)
2. [Run personalized experiments](#2-run-personalized-experiments)
   - 2.a [Run code for setup 1](#2a-run-code-for-setup-1)
   - 2.b [Run code for setup 2](#2a-run-code-for-setup-2)
3. [Installing extra datasets](#3-installing-extra-datasets)

## 1. Reproduce results

All plots and results presented in the submission can be reproduced with a single command:
```
main.sh
```

By default, code to run `Twitter` and `Uber` is commented out. Details on how to install these datasets are available in [3. Installing extra datasets](#3-installing-extra-datasets).

## 2. Run personalized experiments

Here, we detail how to replicate results for setup 1 and 2. Once `Twitter` and `Uber` datasets are installed, setup 3 and scalability can be reproduced using respectivly `runExperimentalSetup3.sh` and `runScalability.sh`.

### 2.a Run code for setup 1

There are two possibilities:

- Through the command line
   ```
   python3 src/setup1.py $dataset $historical $k 
   ```
   - Available datasets include `Abalone`, `OnlineRetail`, `Electricity` (If installed, also `Twitter` and `Uber`).
   - Available algorithms for the historical clustering includes `FFT`, `Resilient`, `Carv`
   - `k` is an integer representing the number of cluster centers.

   Usecase example:
   ```
   python3 src/setup1.py Abalone FFT 20
   ```


- Through the bash script
   ```
   ./runExperimentalSetup1.sh
   ```
   This command creates all results and plots files present in the paper.

Results will be stored in the corresponding `results/setup1/` and `plots/setup1/` folder.
___

### 2.b Run code for second experimental setup

Similarily, there are two possibilities:

- Through the command line 
   ```
   python3 src/setup2.py $dataset $historical $k
   ```
   - Available datasets include `OnlineRetail`, `Electricity` (If installed, also `Twitter` and `Uber`).
   - Available algorithms for the historical clustering includes `FFT`, `Resilient`, `Carv`
   - `k` is an integer representing the number of cluster centers.

   Usecase example:
   ```
   python3 src/setup2.py OnlineRetail Resilient 10
   ```

- To re-create the results from the paper, you may run 
   ```
   ./runExperimentalSetup2.sh
   ```
   This command creates all results and plots files present in the paper.

Results will be stored in the corresponding `results/setup2/` and `plots/setup2/` folder.
___


# 3. Installing extra datasets

## 3.a Uber dataset

Uber dataset is available [here](https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city). Download the archive and place `uber-raw-data-jun14.csv` in `datasets/uber`. Then, comment out ` uberTwoDaysPreprocess()` from `src/createcsv.py` and run `python3 src/createcsv.py`.

## 3.b Twitter dataset

Twitter dataset is a UCI dataset not available through python import. Hence, it has to be manually downloaded [here](https://archive.ics.uci.edu/dataset/1050/twitter+geospatial+data).

Then, decompress the zip file and place the CSV file in `datasets/twitter/twitter.csv`.

___

Contact address: Thibault Marette (marette@kth.se)

README last updated on 16/12/2025.


