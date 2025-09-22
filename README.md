# The reposity for **OMNIXNET:HIGH-RESOLUTIONFY-4BRECONSTRUCTIONVIACROSS-SATELLITE SPECTRALMAPPING**

![model](../OMNIXNET/assets/model.png)

# usage

## step 1

install the requirement dependencies
`pip install -r requirement.txt`

## step 2

inplement your own dataset in ./OMNIXNET/data_privider/data_loader.py and register the dataset in ./OMNIXNET/data_privider/data_factory.py

## step 3

inplement your own model in ./OMNIXNET/models and register the model class in ./OMNIXNET/exp/exp_basic.py

or just use our OMNIXNET

## step 4

run the ./OMNIXNET/scripts/omnixnet.sh or your own scripts
