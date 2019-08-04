import pandas as pd
import numpy as np
import csv
import os



def readcsv(filename):
    data = pd.read_csv(filename) #Please add four spaces here before this line
    return(np.array(data)) #Please add four spaces here before this line


def main():
    # yourArray = readcsv("/home/hamimart/Downloads/TrackingNet 2.0 Test Set Extension - Snowboard(1).tsv")
    # for i in range(30):
    #     print(yourArray[i])

    df = pd.DataFrame.from_csv("/home/hamimart/Downloads/TrackingNet 2.0 Test Set Extension - Ball(5).tsv", sep="\t")
    my_array=np.array(df)
    for i in range(13,15):
        print(f"python main.py --YT_ID {str(my_array[i][0])} --start {int(int(my_array[i][1])/1000)}  --duration {int(my_array[i][8])} --remove_exist --ID {int(my_array[i][4])}")
        os.system(f"python main.py --YT_ID {str(my_array[i][0])} --start {int(int(my_array[i][1])/1000)}  --duration {int(my_array[i][8])} --remove_exist --ID {int(my_array[i][4])}")




if __name__ == '__main__':
    main()