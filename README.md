# DFBT
(D)ownload videos from YT, extract it into (F)rames, draw an intital (B)ounding box and test (T)rackers on it. 

# Requirments:
1. A computer with any kind of linux system to be able to use the os.system
2. Pytracking
3. Pysot
3.1. CUDA10


# what to modify in the code:
After downloading *pytracking* and *pysot*:

Make sure you write down the path of pytracking in line #16 and for pysot in line #23.


# How to run the file:
`python try2.py --YT_ID ID of the YT --start start time in seconds --duration duration in seconds`

example:

`python try2.py --YT_ID ez-F6Qt3Of0 --start 252 --duration 13`

# Problem might occured
In pytracking:
Atom model will have 0 bytes check in pytracking/pytracking/networks/atom_default.pth 
  *download the new model of ATOM from issue 41*
