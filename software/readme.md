# Pre-trained Language Model with Prompts for Temporal Knowledge Graph Completion
### 1.Put _data_ folder into this folder
### 2. Download the pre-trained model
For each folder in _data_, you can download _bert-base-cased.bin_ for pytorch and put it into the corresponding folder.
### 3. Run the code for training
```bash
python PPT.py --dataset ICEWS14 --gpu 0 --max_sample 12 --seq_len 256 --m train --batch_size 32 --mi 0 --max_epochs 5
python PPT.py --dataset ICEWS18 --gpu 0 --max_sample 16 --seq_len 256 --m train --batch_size 32 --mi 0 --max_epochs 5
python PPT.py --dataset ICEWS05 --gpu 0 --max_sample 16 --seq_len 256 --m train --batch_size 32 --mi 0 --max_epochs 5
```
### 4. Run the code for evaluation
```bash
python PPT.py --dataset ICEWS14 --gpu 0 --max_sample 12 --seq_len 256 --m eval --batch_size 32 --mi 0 --epoch 0
python PPT.py --dataset ICEWS18 --gpu 0 --max_sample 16 --seq_len 256 --m eval --batch_size 32 --mi 0 --epoch 0
python PPT.py --dataset ICEWS05 --gpu 0 --max_sample 16 --seq_len 256 --m eval --batch_size 32 --mi 0 --epoch 0
```
### 5. Run the code for testing
```bash
python PPT.py --dataset ICEWS14 --gpu 0 --max_sample 12 --seq_len 256 --m test --batch_size 32 --mi 0 --epoch 0
python PPT.py --dataset ICEWS18 --gpu 0 --max_sample 16 --seq_len 256 --m test --batch_size 32 --mi 0 --epoch 0
python PPT.py --dataset ICEWS05 --gpu 0 --max_sample 16 --seq_len 256 --m test --batch_size 32 --mi 0 --epoch 0
```