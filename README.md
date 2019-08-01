# DFBT
(D)ownload videos from YT, extract it into (F)rames, draw an intital (B)ounding box and test (T)rackers on it. 

# Requirments:
1. A computer with any kind of linux system to be able to use the os.system
2. Pytracking
3. Pysot
4. CUDA 10


# what to modify in the code:

create the conda environment:

`conda env create -f environment.yml`

`conda activate dfb`


Download *pytracking* and *pysot*:

pytracking:

```
git clone https://github.com/visionml/pytracking.git
cd pytracking
git submodule update --init
bash install.sh conda_install_path pytracking
```


pysot:

```
git clone https://github.com/STVIR/pysot.git
cd pysot
python setup.py build_ext --inplace
```
Download the models [here](https://github.com/STVIR/pysot/blob/master/MODEL_ZOO.md)


After downloading *pytracking* and *pysot*:

Make sure you write down the path in Line 158


# How to run the file:
`python main.py --YT_ID <ID of the YT> --start <start time in seconds> --duration <duration in seconds> --path <where to save that sequence> `

example:

`python main.py --YT_ID ez-F6Qt3Of0 --start 252 --duration 13 --path /home/hamimart/Documents/Videos`

# Problem that might occured
In pytracking:

Atom model will have 0 bytes check in pytracking/pytracking/networks/atom_default.pth 

  *download the new model of ATOM from issue 41 in pytracking github page*
