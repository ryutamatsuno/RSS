# Improved mixing time of k-subgraph

Basic implementation of RSS and RSS+ for SDM 2020 paper: Improved mixing time of k-subgraph.



## Dependencies

Please run setup.sh to install numpy, networkx, and pandas by pip.
If you use other package managers, e.g., pip3, conda, install these libraries by mannually.

Python>=3.0
Numpy>=1.16.0
NetworkX>= 2.0
pandas>=0.25.0



## Usage Example

### Actual sampling time

```
> python3 exp_samplingtime.py ba10 3 RSS 100
data set: ba10.edg
n= 10 , m= 16 , k= 3 , e= 0.05
model_name: RSS2
n_samples: 100
      0/100   0.00051570[s]  sample: (1, 3, 4)
     10/100   0.00038052[s]  sample: (2, 4, 5)
     20/100   0.00037479[s]  sample: (1, 4, 5)
     30/100   0.00042248[s]  sample: (2, 7, 8)
     40/100   0.00036335[s]  sample: (4, 5, 9)
     50/100   0.00050211[s]  sample: (2, 3, 6)
     60/100   0.00051332[s]  sample: (2, 3, 7)
     70/100   0.00047135[s]  sample: (1, 4, 9)
     80/100   0.00100541[s]  sample: (2, 4, 8)
     90/100   0.00046992[s]  sample: (2, 3, 4)
Sampling time: 0.0005170845985412598  +- 0.0002009598472167978 [s]
```

### Estimated sampling time

```
> python3 exp_estimatedtime.py ba10 3 RSS 100
data set: ba10
n= 10 , m= 16 , k= 3 , e= 0.05
model_name: RSS
n_samples: 100
Preloading: 3
UniformSampling   : 0.0009583020210266113
DegreePropSampling: 0.38076204110383993

Done: 0.0016303062438964844 [s]
      0/100           0.000638[s]
     10/100           0.001529[s]
     20/100           0.003451[s]
     30/100           0.000499[s]
     40/100           0.002074[s]
     50/100           0.000943[s]
     60/100           0.000419[s]
     70/100           0.001246[s]
     80/100           0.000377[s]
     90/100           0.001246[s]
Estimated Sampling time: 0.0008749151229858398  +- 0.0006712058020837339 [s]
```


## Citation

Will be updated.