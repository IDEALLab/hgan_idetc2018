# Hierarchical-GAN

Please check out our more recent work on this topic [here](https://github.com/IDEALLab/hgan_jmd_2019).

![Alt text](/architecture.png)

## License
This code is licensed under the MIT license. Feel free to use all or portions for your research or related projects so long as you provide the following citation information:

Chen W, Jeyaseelan A, Fuge M. Synthesizing Designs With Inter-Part Dependencies Using Hierarchical Generative Adversarial Networks. ASME. International Design Engineering Technical Conferences and Computers and Information in Engineering Conference, Volume 2A: 44th Design Automation Conference:V02AT03A007. doi:10.1115/DETC2018-85339.

    @inproceedings{chen2018hgan,
        author={Chen, Wei and Jeyaseelan, Ashwin and Fuge, Mark},
        title={Synthesizing Designs with Inter-part Dependencies Using Hierarchical Generative Adversarial Networks},
        booktitle={ASME 2018 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference},
        year={2018},
        month={Aug},
        publisher={ASME},
        address={Quebec City, Canada}
    }

## Required packages

- keras
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

