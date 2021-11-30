# Autoencoder ICP
<p>
David Sebesta

ECE 5258 Pattern Recognition

Fall 2021
</p>




## Purpose
<purpose>
The goal of this project was to create a simple autoencoder using primary component analysis (PCA).
Then using the autoencoder, three sets of data would be encoded and decoded.
The data consists of the mnist 8x8 digits, mnist 28x28 digits, and a cropped version of the Yale Faces.
This autoencoder also allows for the data to be corrupted by noise and reconstructed.
</purpose>


## Technique
<technique>
An basic autoencoder consists of two parts: an encoder and decoder.
The encoder transforms the data to a smaller dimension based on the most important components.
Then the decoder transforms it back to the original dimensions.
The encoding function maps D-dimensions to H-dimensions.
<img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7Bf%3A%5Cmathbb%7BR%7D%5E%7BD%7D%5Cto%5Cmathbb%7BR%7D%5E%7BH%7D%7D">,

The decoding function maps H-dimensions to D-dimensions.
<img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7Bg%20%3A%20%5Cmathbb%7BR%7D%5E%7BH%7D%20%5Cto%20%5Cmathbb%7BR%7D%5E%7BD%7D%7D">

The reconstructed samples after being encoded then decoded, are approximately equal to the original samples.
<img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7B%5Chat%7Bx%7D%20%5Ctriangleq%20g(f(x))%20%5Capprox%20x%7D">

</technique>

### Encoder
<encoder>
The encoder first uses PCA to find the transformation. With training samples <img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7BX%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BRxD%7D%7D">
and total variance explained <img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7B0%20%5Cle%20p%20%5Cle%201%7D">

#### PCA Training
First the sample mean is calculated.
<br />
<img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7B%5Chat%7B%5Cmu%7D_x%20%3D%20%5Cfrac%7B1%7D%7BN%7DX%5E%7BT%7D1_%7BN%7D%7D">

Then the samples are centered.
<br />
<img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7B%5Ctilde%7BX%7D%20%3D%20X%20-%201_N%5Chat%7B%5Cmu%7D_x%7D">

The sample covariance matrix computed using the centered samples.
<br />
<img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7B%5Chat%7BC%7D_%7BX%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Ctilde%7BX%7D%5E%7BT%7D%5Ctilde%7BX%7D%7D">

Eigen Value Decomposition is performed on the sample covariance matrix. The eigen-pairs must be sorted in descending order.
<br />
<img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7B%5B%5Cnu%2C%20%5Clambda%5D%20%5Cquad%20%5Cunderleftarrow%7BEVD%7D%20%5Cquad%20%5Chat%7BC%7D_%7BX%7D%7D">

Find the number of principle components needed to be used to contain the correct amount of total variance.
<br />
<img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7BH%20%3D%20arg%5Cmin%5C%7Bk%5Cin%5C%7B1%2C...%2CD%5C%7D%3A%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%20%5C%20%5Clambda_%7Bi%7D%20%5Cgeq%20p%5C%3Btrace%5C%7B%5Chat%7BC%7D_%7Bx%7D%5C%7D%5C%7D%7D">

#### Encoding Test Samples
The test samples, <img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7BX_t%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BMxD%7D%7D">,
can be easily transformed into their PCA representation, <img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7B%5Ctilde%7BY%7D_t%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BMxH%7D%7D">.

First center the test samples with the mean of the training samples.
<br />
<img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7B%5Ctilde%7BX%7D_t%20%3D%20X_t%20-%201_%7BM%7D%5Chat%7B%5Cmu%7D%5E%7BT%7D_%7Bx%7D%7D">

Transform the centered data using the first H primary components.
<br />
<img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7B%5Ctilde%7BY%7D_t%20%3D%20%5Ctilde%7BX%7D_t%5Cnu_%7BH%7D%7D">

</encoder>



### Decoder
<decoder>
Decoding is very similar to encoding. It uses the same PCA-transformation, but in the other direction mapping H-dimensions back to D-dimensions.

The decoded test samples are caculated using the same H principle components, and then uncentering samples with the training samples mean.
<br />
<img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7B%5Chat%7BX%7D_%7Bt%7D%3D%5Ctilde%7BY%7D_%7Bt%7D%5Cnu%5E%7BT%7D_%7BH%7D%20%2B%5Chat%7B%5Cmu%7D_%7Bx%7D%7D">

</decoder>

## Results

<results>

For the data used in this project, N is the number of samples and D is the number of pixels.
Each dimensions value is equal to the pixels strength from 0 to 255.
That value is then normalized to be between 0 and 1.
For the 8x8 Bit Digits, D = 64.
For the 28x28 Bit Digits, D = 784.
For the Yale Faces, they are normally 200x200, but they have to be downsampled unless you have a very powerful computer.

</results>

### 8x8 Bit Digits
<eight>
With the total variance explained, p, equal to 0.8. The total number of principle components is 13.

![plot](results/8x8/five_p_0.8.png?raw=true )

![plot](results/8x8/three_p_0.8.png?raw=true )

With the total variance explained, p, equal to 0.85. The total number of principle components is 17.

![plot](results/8x8/one_p_0.85.png?raw=true )

![plot](results/8x8/three_p_0.85.png?raw=true )

With the total variance explained, p, equal to 0.9. The total number of principle components is 21.

![plot](results/8x8/two_p_0.9.png?raw=true )

![plot](results/8x8/three_p_0.9.png?raw=true )

With the total variance explained, p, equal to 0.95. The total number of principle components is 29.

![plot](results/8x8/six_p_0.95.png?raw=true )

![plot](results/8x8/one_p_0.95.png?raw=true )

With the total variance explained, p, equal to 0.99. The total number of principle components is 41.

![plot](results/8x8/five_p_0.99.png?raw=true )

![plot](results/8x8/two_p_0.99.png?raw=true )

#### Noise

With the total variance explained, p, equal to 0.9.

![plot](results/8x8/noise_zero_p_0.9.png?raw=true )

With the total variance explained, p, equal to 0.95.

![plot](results/8x8/noise_one_p_0.95.png?raw=true )


The Cumulative Total Variance Explained vs the Number of Components.
See that 90% is at N=21.
Since the other data has a lot more dimensions, the graph gets way to busy to read.

![plot](results/8x8/cve.png?raw=true "Cumulative Variance Explained vs Number of Principle Components")

</eight>

### 28x28 Bit Digits
<twentyeight>

With the total variance explained, p, equal to 0.8. The total number of principle components is 44.

p = 0.85, H = 59

p = 0.9, H = 87

p = 0.95, H = 154

p = 0.99, H = 331

Noise

p = 0.8

p = 0.9

p = 0.95

</twentyeight>

### Yale Faces
<faces>

The faces have been downsampled by 4, so they are 50x50 with D = 2500.
Since all the faces are not centered, the PCA gets slightly off due to the position of the faces.

p = 0.8, H = 11

p = 0.85, H = 17

p = 0.9, H = 26

p = 0.95, H = 50

p = 0.99, H = 106



</faces>