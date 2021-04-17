This is implemented by LMC-ZC.  

You shold first run `preprocess.py` to gain the processed dataset.  
Then run `main.py` to train and test model.  
If you want to change the parameters, you can slightly modified the `parser.py`.  

training detail are following:
```
(spent time: 16.514949798583984s)
epoch: 0, loss = [0.655248 == 0.655180 + 0.000069]
(spent time: 6.206027984619141s)
loss = [0.621406 == 0.621222 + 0.000185], auc = [0.735132]


(spent time: 14.878135681152344s)
epoch: 1, loss = [0.598435 == 0.598078 + 0.000357]
(spent time: 5.728532791137695s)
loss = [0.580817 == 0.580281 + 0.000536], auc = [0.770470]


(spent time: 15.116863012313843s)
epoch: 2, loss = [0.567688 == 0.566975 + 0.000713]
(spent time: 5.975045680999756s)
loss = [0.561489 == 0.560610 + 0.000880], auc = [0.780752]
```
