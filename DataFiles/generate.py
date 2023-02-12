import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import rioxarray as rxr
import torch
import sys

#print sys.argv[0] # prints python_script.py
date = sys.argv[1]
#print sys.argv[2] # prints var2



# Fetching the stable data
overfolder = "./Data2021"
data = []
tmparray = np.zeros((1, 3, 12, 256, 256))
i = 0
altitude = np.load("./TestData/Altitude.npy")
altitude = np.round(altitude).astype("int16")
lsm = np.load("./TestData/LandSea.npy")
lsm = np.round(lsm).astype("int16")


# Looping over the files-in specific date-folder
for folder in sorted(os.listdir(overfolder)):
    if folder == date:
        for file in sorted(os.listdir(overfolder+"/"+folder)):
            try:
                with xr.open_dataset(overfolder+"/"+folder+"/"+file) as dat:
                    dat.load()
                np_data = dat.cma[168+12-40:368-12+40,370+12-40:570-12+40].to_numpy()
                tmparray[0,0,i,]=np_data
                tmparray[0,1,i,]=altitude
                tmparray[0,2,i,]=lsm
                if i == 11:
                    data.append(tmparray)
                    i = 0
                else:
                    i += 1
            except:
                with open("ErrorLog.txt", "a") as log:
                    log.write("Error with: "+folder+", "+file+"\n")
                
# Saving the done-file
cdata = np.concatenate(data).astype("int16")
final = torch.from_numpy(cdata)
torch.save(final, "./PytorchFiles/"+date+".pt")
            
