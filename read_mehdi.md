# For installation, the following steps were done:

There was already a conda env for Monai and this packages was installed in that conda env.
```
git clone https://github.com/jenspetersen/probabilistic-unet.git
pip install -e probabilistic-unet
```
then
```
git clone https://github.com/MIC-DKFZ/trixi.git
cd trixi
# remove the scikit learn versoin
pip install -e .
```
then
```
pip install --upgrade batchgenerators
```

modify the subclasses
```
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
```
