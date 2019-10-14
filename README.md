# Improved mixing time of k-subgraph

Basic implementation of RSS and RSS+ for SDM 2020 paper: Improved mixing time of k-subgraph.



## Dependencies

Please install numpy, networkx, and pandas manually.

- Python >=3.0
- NumPy >=1.16.0
- NetworkX >= 2.0
- pandas >=0.25.0
=

## Note

We keep making the algorithms faster and faster by optimizing calculation. Hence the programs run faster than the times reported in the paper.
 The codes here, which is the latest version, are the best implementation so far. Also, the results highly depends on machine specs.


## Usage Example

### Obtain k-subgrah samples

```
> python3 main.py ba100 5 RSS2 0.01 0.05 100
arguments;
data set         : ba100
k                : 5
model_name       : RSS2
mixing_time_ratio: 0.01
e                : 0.05
n_samples        : 100
n= 100 m= 196  k= 5
    100/100 46.80[s] estimated: 46.80[s]
over all time:46.80[s]
Obtained 5-subgraphs
(4, 9, 12, 41, 70)
(1, 2, 7, 31, 36)
(0, 3, 4, 38, 62)
(3, 4, 5, 28, 45)
(3, 9, 16, 32, 92)
....etc
```

## Experiments

### Uniformity

```
> python3 exp_uniformity.py soc-karate 4 RSS2 0.01 0.05 1000
arguments;
data set         : soc-karate
k                : 4
model_name       : RSS2
mixing_time_ratio: 0.01
e                : 0.05
generating_ratio : 1000
n= 34 m= 78  k= 4
actual number of k-subgraph: 2363
n_samples: 2363000
pre-loading: 2
...time:   0.002[s]
pre-loading: 3
...time:   0.161[s]
pre-loading: 4
...time:   2.023[s]
pi: 0.00042319085907744394
 386245/2363000  2[m]59.87[s] estimated: 18[m]20.41[s] loss:0.03132
 772328/2363000  5[m]58.04[s] estimated: 18[m]15.47[s] loss:0.02216
1157518/2363000  8[m]55.67[s] estimated: 18[m]13.54[s] loss:0.01786
1544243/2363000 11[m]55.24[s] estimated: 18[m]14.46[s] loss:0.01553
1930269/2363000 14[m]53.97[s] estimated: 18[m]14.38[s] loss:0.01383
2316276/2363000 17[m]54.21[s] estimated: 18[m]15.88[s] loss:0.01261
2363000/2363000 18[m]28.69[s] estimated: 18[m]28.69[s] loss:0.01243
over all time:18[m]28.69[s]
loss: 0.012426576385950064
should be smaller than e: 0.05

```

### Actual sampling time

```
> python3 exp_samplingtime.py ba100 5 RSS2 0.01 0.05 100
arguments;
data set         : ba100
k                : 5
model_name       : RSS2
mixing_time_ratio: 0.01
e                : 0.05
n_samples        : 100
n= 100 , m= 196 , k= 5 , e= 0.05
      0/100   0.11357760[s]
     10/100   0.08788133[s]
     20/100   0.13831544[s]
     30/100   0.42238927[s]
     40/100   0.11166596[s]
     50/100   0.20234227[s]
     60/100   0.10551357[s]
     70/100   1.28051376[s]
     80/100   1.41313481[s]
     90/100   0.58689308[s]
Sampling time: 0.3596782636642456  +- 0.303922752851285 [s]
 ~   0.36[s]
```

### Estimated sampling time

```
> python3 exp_estimatedtime.py ba1000 5 RSS2 0.01 0.05 100
arguments;
data set         : ba1000
k                : 5
model_name       : RSS2
mixing_time_ratio: 0.01
e                : 0.05
n_samples        : 100
n= 1000 , m= 1996 , k= 5 , e= 0.05
Preloading: 3
UniformSampling(3)   : 0.00025741815567016603
DegreePropSampling(3): 0.022935573291778564
Done: 0.0015039443969726562 [s]
Preloading: 4
UniformSampling(4)   : 0.03597504796981812
DegreePropSampling(4): 2.7592527645092013
Done: 0.0008516311645507812 [s]
      0/100           2.786326[s]
     10/100           2.774249[s]
     20/100           8.284748[s]
     30/100          13.834887[s]
     40/100           2.786267[s]
     50/100          19.326597[s]
     60/100           2.763487[s]
     70/100           2.744342[s]
     80/100           5.510705[s]
     90/100           8.303727[s]
Estimated Sampling time: 7.3951951717948905  +- 5.390041545307455 [s]
 ~   7.40[s]
```


## Citation

If accepted.

## Licence

Will be updated.