This is implemented by LMC_ZC.  
  

The experiments contains many models include BPRMF, NGCF, LR_GCCF, LightGCN.  
  
  
We consider *gowalla* dataset as an example:  
you should first run `data/preprocess.py` to process raw data, then run `main/*` to train model. also,you need to change the path in `main/*` to your own.  
  
  
folder `log` recoder the the training process, for example:  
```
epoch:1, loss:[0.438004] = mf:[0.435858] + reg:[0.002147]
recall@10:[0.059646], ndcg@10:[0.058437], recall@20:[0.086229], ndcg@20:[0.070056]
epoch:2, loss:[0.205175] = mf:[0.199136] + reg:[0.006038]
recall@10:[0.068120], ndcg@10:[0.068062], recall@20:[0.096960], ndcg@20:[0.080796]
epoch:3, loss:[0.161834] = mf:[0.154317] + reg:[0.007517]
recall@10:[0.072840], ndcg@10:[0.073122], recall@20:[0.103551], ndcg@20:[0.086612]
epoch:4, loss:[0.138590] = mf:[0.130175] + reg:[0.008415]
recall@10:[0.076528], ndcg@10:[0.077538], recall@20:[0.107594], ndcg@20:[0.091219]
epoch:5, loss:[0.123620] = mf:[0.114542] + reg:[0.009078]
recall@10:[0.079100], ndcg@10:[0.080552], recall@20:[0.111215], ndcg@20:[0.094694]
epoch:6, loss:[0.114332] = mf:[0.104702] + reg:[0.009629]
recall@10:[0.080598], ndcg@10:[0.082303], recall@20:[0.113816], ndcg@20:[0.096863]
epoch:7, loss:[0.106706] = mf:[0.096659] + reg:[0.010047]
recall@10:[0.082520], ndcg@10:[0.084113], recall@20:[0.116584], ndcg@20:[0.099082]
epoch:8, loss:[0.100404] = mf:[0.089983] + reg:[0.010422]
recall@10:[0.084101], ndcg@10:[0.085851], recall@20:[0.118445], ndcg@20:[0.100954]
epoch:9, loss:[0.095944] = mf:[0.085194] + reg:[0.010750]
recall@10:[0.085317], ndcg@10:[0.086943], recall@20:[0.119981], ndcg@20:[0.102201]
epoch:10, loss:[0.091665] = mf:[0.080619] + reg:[0.011046]
recall@10:[0.086336], ndcg@10:[0.088031], recall@20:[0.121370], ndcg@20:[0.103435]
epoch:11, loss:[0.087468] = mf:[0.076153] + reg:[0.011315]
recall@10:[0.087354], ndcg@10:[0.088973], recall@20:[0.122276], ndcg@20:[0.104389]
epoch:12, loss:[0.084525] = mf:[0.072950] + reg:[0.011576]
recall@10:[0.087839], ndcg@10:[0.089758], recall@20:[0.123654], ndcg@20:[0.105519]
epoch:13, loss:[0.081718] = mf:[0.069923] + reg:[0.011794]
recall@10:[0.089092], ndcg@10:[0.090636], recall@20:[0.125482], ndcg@20:[0.106562]
epoch:14, loss:[0.079191] = mf:[0.067201] + reg:[0.011990]
recall@10:[0.089653], ndcg@10:[0.091414], recall@20:[0.126482], ndcg@20:[0.107557]
```
