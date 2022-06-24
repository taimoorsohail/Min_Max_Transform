## Import the BSP component of the WM_Methods package
from WM_Methods import BSP
## Other required packages for calculations and plotting
import numpy as np
import xarray as xr
## This package allows for communication between the submission script and this python script
import sys
import matplotlib.pyplot as plt


## Load the ACCESS temperature, salinity and volume:
chunks = {'time': 1}
# Area
ACCESS_areacello_piControl = xr.open_mfdataset('/g/data/e14/txs156/Data/CMIP/ACCESS-CM2/3D_files/piControl/areacello_Ofx_ACCESS-CM2_piControl_r1i1p1f1_gn.nc', decode_times=True, chunks = chunks)
# Volume
ACCESS_volcello_piControl = xr.open_mfdataset('/g/data/e14/txs156/Data/CMIP/ACCESS-CM2/3D_files/piControl/volcello_Omon_ACCESS-CM2_piControl_r1i1p1f1_gn_*', decode_times=True, chunks = chunks)
# Temperature
ACCESS_bigthetao_piControl = xr.open_mfdataset('/g/data/e14/txs156/Data/CMIP/ACCESS-CM2/3D_files/piControl/bigthetao_Omon_ACCESS-CM2_piControl_r1i1p1f1_gn_*', decode_times=True, chunks = chunks)
# Salinity
ACCESS_so_piControl = xr.open_mfdataset('/g/data/e14/txs156/Data/CMIP/ACCESS-CM2/3D_files/piControl/so_Omon_ACCESS-CM2_piControl_r1i1p1f1_gn_*', decode_times=True, chunks = chunks)

## Load the ACCESS surface fluxes:
ACCESS_wfo_piControl_files = xr.open_mfdataset('/g/data/e14/txs156/Data/CMIP/ACCESS-CM2/3D_files/piControl/wfo*',decode_times=True, chunks = chunks) 
ACCESS_hfds_piControl_files = xr.open_mfdataset('/g/data/e14/txs156/Data/CMIP/ACCESS-CM2/3D_files/piControl/hfds*',decode_times=True, chunks = chunks) ##convert to W and kgs^-1, respectively

ACCESS_wfo_piControl = ACCESS_wfo_piControl_files.wfo*ACCESS_areacello_piControl.areacello ##convert to kgs^-1
ACCESS_hfds_piControl = ACCESS_hfds_piControl_files.hfds*ACCESS_areacello_piControl.areacello ##convert to W

## Load the basins mask which will be used to split the Basins up:
ACCESS_mask = xr.open_mfdataset('/g/data/e14/txs156/Analysis/Min_Transform_Method/Mask/mask_ACCESS.nc',decode_times=True, chunks = chunks)
land_mask = (ACCESS_hfds_piControl_files.hfds.isel(time=0)/ACCESS_hfds_piControl_files.hfds.isel(time=0))

## Ensure flattened array shapes align by adding a time axis to the mask, and depth axis to area:
ACCESS_lat = ACCESS_volcello_piControl.latitude.values
ACCESS_lon = ACCESS_volcello_piControl.longitude.values
ACCESS_depth = ACCESS_volcello_piControl.lev.values

dArea_4D = ACCESS_areacello_piControl.areacello.expand_dims({'lev':ACCESS_depth.size},axis=0).assign_coords(lev=ACCESS_depth)
hfds_4D = ACCESS_hfds_piControl.expand_dims({'lev':ACCESS_depth.size},axis=1).assign_coords(lev=ACCESS_depth)
wfo_4D = ACCESS_wfo_piControl.expand_dims({'lev':ACCESS_depth.size},axis=1).assign_coords(lev=ACCESS_depth)

# Set surface area to be 0 everywhere except the surface
dArea_4D[1:,:,:] = 0
# Set surface fluxes to be 0 everywhere except the surface
hfds_4D[:,1:,:,:] = 0
wfo_4D[:,1:,:,:] = 0

## Define basin names:
Basins = ACCESS_mask.Basins.values

## Define the number of BSP bins to output, where number = 2**tree_depth:
tree_depth = 7

## Define a new array 'volcut', which only considers the ocean volume shallower than xxx m. 
# This will be used for our BSP binning as the variable 'v':
depth_cut = 2000
volcut = ACCESS_volcello_piControl.volcello.copy(deep=True)
depth_ind = np.argmin(ACCESS_depth<depth_cut)
volcut[:,depth_ind:,:,:] = 0

## Flatten all arrays of interest
mask_flattened = (ACCESS_mask.mask_ACCESS.stack(z=("i", "j", "lev")))
volcello_flattened = ACCESS_volcello_piControl.volcello.stack(z=("i", "j", "lev"))
areacello_flattened = dArea_4D.stack(z=("i", "j", "lev"))
volcut_flattened = (volcut.stack(z=("i", "j", "lev")))
bigthetao_flattened = (ACCESS_bigthetao_piControl.bigthetao.stack(z=("i", "j", "lev")))
so_flattened = (ACCESS_so_piControl.so.stack(z=("i", "j", "lev")))

## Including ACCESS surface fluxes
hfds_flattened = hfds_4D.stack(z=("i", "j", "lev"))
wfo_flattened = wfo_4D.stack(z=("i", "j", "lev"))

# # Calculate the BSP bins in a time loop:
# # For gadi, we run multiple jobs which each cover a certain time window of the total time series. 
# # Here, we define the time window, 'window', and read the window #, 'ti'

window = 60
ti = int(sys.argv[1]) # This is provided by the submission script

## Define empty arrays that will be filled by the loop BSP function output
partitions = np.zeros((Basins.size, window, 2**tree_depth, 4))
T_mean = np.zeros((Basins.size, window, 2**tree_depth))
S_mean = np.zeros((Basins.size, window, 2**tree_depth))
V_sum = np.zeros((Basins.size, window, 2**tree_depth))
A_sum = np.zeros((Basins.size, window, 2**tree_depth))
hfds_sum = np.zeros((Basins.size, window, 2**tree_depth))
wfo_sum = np.zeros((Basins.size, window, 2**tree_depth))

time_array = np.zeros(window)
## Run the time and basin loop
for i in range(ti*window, (ti+1)*window):
    time_array[int(i-ti*window)] = i#np.floor(i/12)+months_repeating[int(i-ti*window)]
    for j in range(Basins.size):
        ## Pick out the flattened arrays at time i:
        S = so_flattened[i,:].values
        T = bigthetao_flattened[i,:].values
        VCUT = volcut_flattened[i,:].values
        V = volcello_flattened[i,:].values*mask_flattened[j,:].values
        A = areacello_flattened.values*mask_flattened[j,:].values
        HFDS = hfds_flattened[i,:].values*mask_flattened[j,:].values
        WFO = wfo_flattened[i,:].values*mask_flattened[j,:].values

        # Clean out NAN values
        # Here we assume the NaNs in S are the NaNs in all other arrays, not necessarily true
        idx = np.isfinite(S)
        S = S[idx]
        T = T[idx]
        VCUT = VCUT[idx]
        V = V[idx]
        A = A[idx]
        HFDS = HFDS[idx]
        WFO = WFO[idx]
        
        BSP_out = BSP.calc(S,T,VCUT, depth=tree_depth, axis=1, mean=[S,T],sum=[V,A, HFDS, WFO], weight=V)
       # Split the output into constituent diagnostics
        vals = BSP.split(BSP_out, depth=tree_depth)
        partitions[j,int(i-ti*window),:,:] = vals['bounding_box']
        S_mean[j,int(i-ti*window),:] = vals['meaned_vals'][:,0]
        T_mean[j,int(i-ti*window),:] = vals['meaned_vals'][:,1]
        V_sum[j,int(i-ti*window),:] = vals['summed_vals'][:,0]
        A_sum[j,int(i-ti*window),:] = vals['summed_vals'][:,1]
        hfds_sum[j,int(i-ti*window),:] = vals['summed_vals'][:,2]
        wfo_sum[j,int(i-ti*window),:] = vals['summed_vals'][:,3]

# # Save the outputs to a netcdf file
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
da_hfds_sum = xr.DataArray(data = hfds_sum, dims = ["Basin", "Time","Depth"],
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="Net surface heat flux (ERA5)", units="J", variable_id="Basin V_sum"))
da_wfo_sum = xr.DataArray(data = wfo_sum, dims = ["Basin", "Time","Depth"],
                           coords=dict(Basin = Basins, Time = time_array, Depth=np.arange(2**tree_depth)),
                        attrs=dict(description="Net water flux out (ERA5)", units="kgs^-1", variable_id="EN4 A_sum"))

## Create xarray DataSet that will hold all these DataArrays
ds_BSP = xr.Dataset()
ds_BSP['Partitions'] = da_partitions
ds_BSP['T_mean'] = da_T_mean
ds_BSP['S_mean'] = da_S_mean
ds_BSP['V_sum'] = da_V_sum
ds_BSP['A_sum'] = da_A_sum
ds_BSP['hfds_sum'] = da_hfds_sum
ds_BSP['wfo_sum'] = da_wfo_sum

ds_BSP.to_netcdf('BSP_processed/BSP_ACCESS_TS_%i_%i.nc' %(ti*window, (ti+1)*window-1))
