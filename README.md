# Hierarchical-GAN

![Alt text](/architecture.png)

## Required packages

- tensorflow
- numpy
- matplotlib

## Usage

Generate the dataset of airfoil + hole:

```bash
cd airfoil
python build_data.py
```

Generate the dataset of superformula + ellipse:

```bash
cd superformula
python build_data.py
```

Train/evaluate:

```bash
python main.py
```

positional arguments:
    
```
mode	startover, continue or evaluate
data	airfoil or superformula
model	select which model to use: 2g1d (2G1D), naive (2G2D), or wo_info (2G1D without mutual information loss)
```

optional arguments:

```
-h, --help            	show this help message and exit
--train_steps		training steps
--save_interval 	number of intervals for saving the trained model and plotting results
```

