## Dataset

You can download both the raw and processed versions of the dataset from the following location:

[**Dataset**](https://iis-people.ee.ethz.ch/~datasets/DeepMFminiDataset/)

To download the dataset on a windows machine, follow these steps:
1. Open PowerShell as an Administrator
2. Install Chocolatey: [https://chocolatey.org/install](https://chocolatey.org/install)
3. Install wget: 
```plaintext
choco install wget -y
```
4. Download the dataset (this may take a while depending on your internet connection): 
```plaintext
wget --recursive --no-parent --no-host-directories --cut-dirs=2 --reject="index.html*" https://iis-people.ee.ethz.ch/~datasets/DeepMFminiDataset/Data/
```

To download the dataset on a linux machine, follow these steps:
1. Open a bash shell
2. Install wget: 
```plaintext
sudo apt update && sudo apt install -y wget
```
3. Download the dataset (this may take a while depending on your internet connection): 
```plaintext
wget --recursive --no-parent --no-host-directories --cut-dirs=2 --reject="index.html*" https://iis-people.ee.ethz.ch/~datasets/DeepMFminiDataset/Data/
```

Copy the dataset in the Data folder with the following structure:

```plaintext
Data/
├──Processed/
├	├── S1/
├	├── .../
├	└── SM/
├		├── R1/
├		├── .../
├		└── RN/
├			├── gt.mat
├			├── 1.mat
├			├── ...
├			└── L.mat
└──Raw/
	├── S1/
	├── .../
	└── SM/
		├── Data_YYYYMMDD_HHMMSS.bin
		└── ...
```