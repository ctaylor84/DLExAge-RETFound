## DLExAge-RETFound - A fork of RETFound for predicting age from retinal images

### 🔧Install environment

1. Create environment with conda:

```
conda create -n dlexage python=3.11.8 -y
conda activate dlexage
```

2. Install dependencies

```
git clone https://github.com/ctaylor84/DLExAge-RETFound.git
cd DLExAge-RETFound
pip install -r requirement.txt
```

To use ResNet and EfficientNet v2 models, timm must be updated to version 0.6.5.


### 🌱Model training

To train a regression model, follow these steps:

1. Download the RETFound pre-trained weights
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Large</th>
<!-- TABLE BODY -->
<tr><td align="left">Colour fundus image</td>
<td align="center"><a href="https://drive.google.com/file/d/1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE/view?usp=sharing">download</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">OCT</td>
<td align="center"><a href="https://drive.google.com/file/d/1m6s7QYkjyjJDlpEuXm7Xp3PmjN-elfW2/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

2. Organise your data into this directory structure, with all the images being placed in the 'images' directory.

```
├── datasets
    ├──bb_age_+0_+0
        ├──images
        ├──train.csv
        ├──val.csv
        ├──test.csv
``` 

3. Each csv file requires a "file" column for image file names, and a "target" column containing the regression target value (floating-point). At the moment, only one target value is supported.

4. Start fine-tuning. A fine-tuned checkpoint will be saved during training. Evaluation will be run after training. For example:

```
python main_finetune.py \
    --batch_size 16 \
    --model vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --data_path ./datasets/bb_age_+0_+0/ \
    --task ./finetune/bb_age_+0_+0/ \
    --finetune ./RETFound_oct_weights.pth \
    --input_size 224
```


5. Example for evaluation only:

```
python main_finetune.py \
    --eval --batch_size 16 \
    --model vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --data_path ./datasets/bb_age_+0_+0/ \
    --task ./eval/bb_age_+0_+0/ \
    --resume ./finetune/bb_age_+0_+0/checkpoint-best.pth \
    --input_size 224
```

