# TAE <br>
# Environment <br>
python 3.7 <br>
pytorch 1.8.0 <br>
# Dataset <br>
There are four datasets: YAGO, WIKI, ICEWS14 and ICEWS05-15. Each data folder has 'train.txt', 'valid.txt', 'test.txt'. <br> 
# Run the experiments <br>
1. cd ./TAE/train <br>
2. python train_TAEconve.py (defult setting) 
3. python --dataset YAGO ----lr-conv 0.001 ----time-interval 1 ----n-epochs-conv 50 ----batch-size-conv 50 --pred sub --valid-epoch 5 --count 8 (you can setting parameters this way)
