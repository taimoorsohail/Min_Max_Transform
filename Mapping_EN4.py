## Import the BSP and Remapping components of the WM_Methods package
from WM_Methods import Remapping
## Other required packages for calculations and plotting
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys

## LOAD T and S data from a gridded observations (e.g., we use EN4 here)
data = xr.open_mfdataset('/g/data/e14/txs156/Data/Observations/EN4/EN4_CT_SA_*')
## Load BSP Data
EN4_BSP_data = xr.open_mfdataset('/g/data/e14/txs156/Analysis/Min_Transform_Method/BSP/BSP_processed/BSP_EN4_TS_*.nc')

## Calculate the BSP bins in a time loop:
## For gadi, we run multiple jobs which each cover a certain time window of the total time series. 
## Here, we define the time window, 'window', and read the window #, 'ti'

window = 2
ti = int(sys.argv[1]) # This is provided by the submission script
start = window*ti
end = (ti+1)*window

T = data.Cons_Temp.isel(time=slice(start,end))
S = data.Abs_Sal.isel(time=slice(start,end))

Part_early = EN4_BSP_data.Partitions.isel(Time=slice(start,end)).mean('Basin')

Basins = EN4_BSP_data.Basin.values

fuzz_output = Remapping.remap_mask(T,S, Part_early, depth=int(np.log2(Part_early.shape[-2])), interp=False)

ds_output = xr.Dataset()
ds_output['fuzz'] = fuzz_output

ds_output.to_netcdf('/g/data/e14/txs156/Analysis/Min_Transform_Method/Mapping/Masks_processed/Masks_EN4_%i_%i.nc' %(ti*window, (ti+1)*window-1))

