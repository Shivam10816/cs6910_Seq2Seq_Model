# CS6910 Assignment 3
Assignment 3 submission for the course CS6910 Fundamentals of Deep Learning.

Student Information: Shivam Kharat (CS22M082)

Find Wandb report here : [https://rb.gy/dvqrgv](https://wandb.ai/vilgax/CS6910_Assignment3/reports/ASSIGNMENT-3--Vmlldzo0MzI5OTcy?accessToken=b6218h2p1zh5xkb0scyh3reka1x5ka220tlwcgaurpand2p8ragrsmkn9b91ekfi)
---
Code for EncoderRNN , DecoderRNN and training seq2seq model can be found in file named **Assignment3.ipynb** .
## How to run **Assignment3.ipynb** ? 

-----
For kaggle . If you want to use kaggle comment out following code in Notebook else comment it.
-----
1. Import Notebook to kaggle
2. Add data to folder named dlassgnn
3. make change in data path according to folder
```Python
   #------------------------------------For Kaggle interFace---------------------------------

    self.train_df = pd.read_csv(f"/kaggle/input/dlassgnn/hin_train.csv", header = None)
    self.val_df = pd.read_csv(f"/kaggle/input/dlassgnn/hin_valid.csv", header = None)
    self.test_df = pd.read_csv(f"/kaggle/input/dlassgnn/hin_test.csv", header = None)

```
4. Make changes in wandb configurations(Either in wandb sweep or in Best_Run) as required
5. comment out required code.

```Python
wandb.finish()
wandb_runs(data)
```
or

```Python
wandb.finish()  
Best_Run(data)
```

--------------------------
For Colab
--------------------------
1. Import Notebook to Colab
2. Add aksharantar_sampled folder to google drive in location Mydrive
3. comment out following code
```Python
  #------------------------------------For colab interface-------------------------------------

    self.train_df = pd.read_csv(f"drive/MyDrive/aksharantar_sampled/{lang}/{lang}_train.csv", header = None)
    self.val_df = pd.read_csv(f"drive/MyDrive/aksharantar_sampled/{lang}/{lang}_valid.csv", header = None)
    self.test_df = pd.read_csv(f"drive/MyDrive/aksharantar_sampled/{lang}/{lang}_test.csv", header = None)

```
4. In colab you can load any data you want by providing aprropriate lang name to following code section

```Python
data = EncodedData("hin")
```
5. Make changes in wandb configurations(Either in wandb sweep or in Best_Run) as required
6. comment out required code.

```Python
wandb.finish()
wandb_runs(data)
```
or

```Python
wandb.finish()  
Best_Run(data)
```

------------------------------------------------------------------------

## How to run trainSeq2Seq.py

1. This file will run only in colab
2. To run this file in colab upload this file and Data of hin folder (only csv files)
3. Set accelarator to gpu
4. run following commnads with appropriate parameters
```Python
!pip install wandb
!python trainSeq2Seq.py -key "Wandb_API_Key" -wp 'project_name' -we 'sweep_name' -ct "RNN" -e 30 -b 16 -enc 1 -dec 1 -lr 0.001 -ebs 64 -hs 1024 -drp 0.2 -tr 0.0 -bdr "Yes" -at "No" 
```
5. if you have different file path edit path in following code section

```Python
    self.train_df = pd.read_csv(f"/content/hin_train.csv", header = None)
    self.val_df = pd.read_csv(f"/content/hin_valid.csv", header = None)
    self.test_df = pd.read_csv(f"/content/hin_test.csv", header = None)
```
