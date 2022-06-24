## Import the BSP component of the WM_Methods package
from WM_Methods import BSP
## Other required packages for calculations and plotting
import numpy as np
import xarray as xr
## This package allows for communication between the submission script and this python script
import sys

## Load the EN4 temperature, salinity and volume:
chunks = {'time': 1}
EN4_Data = xr.open_mfdataset('/g/data/e14/txs156/Data/Observations/EN4/EN4_CT_SA_*.nc',decode_times=True, chunks = chunks)

## Load the basins mask which will be used to split the Basins up:
EN4_mask = xr.open_mfdataset('/g/data/e14/txs156/Analysis/Min_Transform_Method/Mask/mask_EN4.nc', decode_times=True, chunks = chunks)

## Ensure flattened array shapes align by adding a time axis to the mask, and depth axis to area:
EN4_depth = EN4_Data.depth.values

EN4_mask_4D = EN4_mask.expand_dims({'time':EN4_Data.time.size},axis=1).assign_coords(time=EN4_Data.time)
dArea_4D = EN4_Data.dArea.expand_dims({'depth':EN4_depth.size},axis=1).assign_coords(depth=EN4_depth)
# Set surface area to be 0 everywhere except the surface
dArea_4D[:,1:,:,:] = 0

## Define basin names:
Basins = EN4_mask.Basins.values

## Define the number of BSP bins to output, where number = 2**tree_depth:
tree_depth = 7

## Define a new array 'volcut', which only considers the ocean volume shallower than xxx m. 
# This will be used for our BSP binning as the variable 'v':
depth_cut = 500
volcut = EN4_Data.dVol.copy(deep=True)
depth_ind = np.argmin(EN4_depth<depth_cut)
volcut[:,depth_ind:,:,:] = 0

## Flatten all arrays of interest
mask_flattened = (EN4_mask_4D.__xarray_dataarray_variable__.stack(z=("lon", "lat", "depth")))
volcello_flattened = EN4_Data.dVol.stack(z=("lon", "lat", "depth"))
areacello_flattened = dArea_4D.stack(z=("lon", "lat", "depth"))
volcut_flattened = (volcut.stack(z=("lon", "lat", "depth")))
bigthetao_flattened = (EN4_Data.Cons_Temp.stack(z=("lon", "lat", "depth")))
so_flattened = (EN4_Data.Abs_Sal.stack(z=("lon", "lat", "depth")))

## Calculate the BSP bins in a time loop:
## For gadi, we run multiple jobs which each cover a certain time window of the total time series. 
## Here, we define the time window, 'window', and read the window #, 'ti'

window = 2
ti = int(sys.argv[1]) # This is provided by the submission script

## Define empty arrays that will be filled by the loop BSP function output
partitions = np.zeros((Basins.size, window, 2**tree_depth, 4))
T_mean = np.zeros((Basins.size, window, 2**tree_depth))
S_mean = np.zeros((Basins.size, window, 2**tree_depth))
V_sum = np.zeros((Basins.size, window, 2**tree_depth))
A_sum = np.zeros((Basins.size, window, 2**tree_depth))
time_array = np.zeros(window)
## Run the time and basin loop
for i in range(ti*window, (ti+1)*window):
    time_array[int(i-ti*window)] = i#np.floor(i/12)+months_repeating[int(i-ti*window)]
    for j in range(Basins.size):
        ## Pick out the flattened arrays at time i:
        S = so_flattened[i,:].values
        T = bigthetao_flattened[i,:].values
        VCUT = volcut_flattened[i,:].values
        V = volcello_flattened[i,:].values*mask_flattened[j,i,:].values
        A = areacello_flattened[i,:].values*mask_flattened[j,i,:].values

        # Clean out NAN values
        # Here we assume the NaNs in S are the NaNs in all other arrays, not necessarily true
        idx = np.isfinite(S)
        S = S[idx]
        T = T[idx]
        VCUT = VCUT[idx]
        V = V[idx]
        A = A[idx]

        BSP_out = BSP.calc(S,T,VCUT, depth=tree_depth, axis=1, mean=[S,T],sum=[V,A],weight=V)
        # Split the output into constituent diagnostics
        vals = BSP.split(BSP_out, depth=tree_depth)
        partitions[j,int(i-ti*window),:,:] = vals['bounding_box']
        S_mean[j,int(i-ti*window),:] = vals['meaned_vals'][:,0]
        T_mean[j,int(i-ti*window),:] = vals['meaned_vals'][:,1]
        V_sum[j,int(i-ti*window),:] = vals['summed_vals'][:,0]
        A_sum[j,int(i-ti*window),:] = vals['summed_vals'][:,1]


## Save the outputs to a netcdf file
# Convert numpy arrays to xarray DataArrays to easily save to netCDF. 

da_partitions = xr.DataArray(data = partitions, dims = ["Basin","Time", "Depth", "Coords"],
                           coords=dict(Basin = Basins, Time = time_array, Depth= np.arange(2**tree_depth), Coords = np.arange(4)),
                        attrs=dict(description="[x0,y0,xmax,ymax] bounds of BSP framework", variable_id="Partitions"))
da_S_mean = xr.DataArray(data = S_mean, dims = ["Basin", "Time", "Depth"],
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="Mean Salinity", units="g/kg", variable_id="EN4 S"))
da_T_mean = xr.DataArray(data = T_mean, dims = ["Basin", "Time","Depth"],
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="Mean Temperature", units="K", variable_id="EN4 T"))
da_V_sum = xr.DataArray(data = V_sum, dims = ["Basin", "Time","Depth"],
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="Total Volume", units="m^3", variable_id="Basin V_sum"))
da_A_sum = xr.DataArray(data = A_sum, dims = ["Basin", "Time","Depth"],
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="Total Area", units="m^2", variable_id="EN4 A_sum"))

## Create xarray DataSet that will hold all these DataArrays
ds_BSP = xr.Dataset()
ds_BSP['Partitions'] = da_partitions
ds_BSP['T_mean'] = da_T_mean
ds_BSP['S_mean'] = da_S_mean
ds_BSP['V_sum'] = da_V_sum
ds_BSP['A_sum'] = da_A_sum

ds_BSP.to_netcdf('/g/data/e14/txs156/Analysis/BSP_processed/BSP_EN4_TS_hist_area_%i_%i.nc' %(ti*window, (ti+1)*window-1))