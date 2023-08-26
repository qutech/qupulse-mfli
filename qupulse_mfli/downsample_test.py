import numpy as np
import qpmfli_downsample_rust
import time

n_windows = np.geomspace(1,10000,num=10,dtype=int)
n_data_points = np.geomspace(1,1e7,num=10,dtype=int)

def benchmark_rust(n_windows,n_data_points,):
    
    time_arr = np.zeros((len(n_windows),len(n_data_points)))
    time_of_trigger = 0.0

    for i, n_w in enumerate(n_windows):
        for j, n_d in enumerate(n_data_points):
            applicable_data = np.linspace(0,1.7,n_d,dtype=np.float32)
            time_data = np.linspace(0.,10,n_d,dtype=np.float32)
            begins = np.linspace(0.,10.,n_w,dtype=np.float32)
            lengths = np.ones(n_w,dtype=np.float32)*0.1
            
            start = time.time()
            result = qpmfli_downsample_rust.py_extract_data(applicable_data, time_data, begins, lengths, time_of_trigger)
            end = time.time()
            time_arr[i,j] = end - start
            
    return time_arr

#%%

time_arr = benchmark_rust(n_windows,n_data_points)


#%%
    # print("hello")
    
    # print(end - start)

n_w = 100
applicable_data = np.linspace(0.0, 1.7, 10001,dtype=np.float32)
time_data = np.linspace(0.0, 1000.0, 10001,dtype=np.float32)
begins = np.linspace(0.0, 1000.0, n_w,dtype=np.float32)
lengths = np.ones(n_w,dtype=np.float32)*0.1
time_of_trigger = 0.0

result = qpmfli_downsample_rust.py_extract_data(applicable_data, time_data, begins, lengths, time_of_trigger)
print(result)

print(result[np.where(~np.isnan(result))].shape)


#%%

'''

results in seconds on threadripper 3975WX 32 cores 100% utilization:

y-axis: windows: array([    1,     2,     7,    21,    59,   166,   464,  1291,  3593,
       10000]) 
x-axis: data_points: array([       1,        5,       35,      215,     1291,     7742,
          46415,   278255,  1668100, 10000000])

[[ 0.     0.     0.     0.     0.     0.     0.001  0.005  0.031  0.184]
 [ 0.001  0.     0.     0.     0.     0.001  0.001  0.006  0.032  0.185]
 [ 0.     0.     0.001  0.     0.001  0.     0.001  0.005  0.032  0.199]
 [ 0.     0.     0.001  0.     0.     0.     0.001  0.006  0.033  0.243]
 [ 0.     0.001  0.     0.     0.     0.     0.002  0.007  0.043  0.25 ]
 [ 0.001  0.     0.001  0.     0.     0.001  0.003  0.015  0.088  0.554]
 [ 0.     0.001  0.     0.001  0.     0.001  0.006  0.036  0.215  1.268]
 [ 0.     0.001  0.     0.001  0.001  0.003  0.016  0.095  0.565  3.409]
 [ 0.001  0.001  0.     0.     0.002  0.007  0.043  0.256  1.535  9.243]
 [ 0.001  0.001  0.001  0.001  0.004  0.02   0.119  0.713  4.25  25.731]]


'''
