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
The encoding function <img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7Bf%3A%5Cmathbb%7BR%7D%5E%7BD%7D%5Cto%5Cmathbb%7BR%7D%5E%7BH%7D%7D">,
maps D-dimensions to H-dimensions.
The decoding function <img src="https://render.githubusercontent.com/render/math?math=g%20%3A%20%5Cmathbb%7BR%7D%5E%7BH%7D%20%5Cto%20%5Cmathbb%7BR%7D%5E%7BD%7D">,
maps H-dimensions to D-dimensions.
The reconstructed samples <img src="https://render.githubusercontent.com/render/math?math=%5Chat%7Bx%7D%20%5Ctriangleq%20g(f(x))%20%5Capprox%20x">,
after being encoded then decoded, are approximately equal to the original samples.
</technique>

### Encoder
<encoder>
The encoder first uses PCA to find the transformation. With training samples <img src="https://render.githubusercontent.com/render/math?math=X%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BRxD%7D">
and total variance explained <img src="https://render.githubusercontent.com/render/math?math=0%20%5Cle%20p%20%5Cle%201">. The
PCA training is as follows.

##### PCA Training
First the sample mean is calculated.
<br />
<img src="https://render.githubusercontent.com/render/math?math=%5Chat%7B%5Cmu%7D_x%20%3D%20%5Cfrac%7B1%7D%7BN%7DX%5E%7BT%7D1_%7BN%7D">

Then the samples are centered.
<br />
<img src="https://render.githubusercontent.com/render/math?math=%5Ctilde%7BX%7D%20%3D%20X%20-%201_N%5Chat%7B%5Cmu%7D_x">

The sample covariance matrix computed using the centered samples.
<br />
<img src="https://render.githubusercontent.com/render/math?math=%5Chat%7BC%7D_%7BX%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Ctilde%7BX%7D%5E%7BT%7D%5Ctilde%7BX%7D">

Eigen Value Decomposition is performed on the sample covariance matrix. The eigen-pairs must be sorted in descending order.
<br />
<img src="https://render.githubusercontent.com/render/math?math=%5B%5Cnu%2C%20%5Clambda%5D%20%5Cquad%20%5Cunderleftarrow%7BEVD%7D%20%5Cquad%20%5Chat%7BC%7D_%7BX%7D">

Find the number of principle components needed to be used to contain the correct amount of total variance.
<br />
<img src="https://render.githubusercontent.com/render/math?math=H%20%3D%20arg%5Cmin%5C%7Bk%5Cin%5C%7B1%2C...%2CD%5C%7D%3A%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%20%5C%20%5Clambda_%7Bi%7D%20%5Cgeq%20p%5C%3Btrace%5C%7B%5Chat%7BC%7D_%7Bx%7D%5C%7D%5C%7D">

##### Encoding Test Samples
The test samples, <img src="https://render.githubusercontent.com/render/math?math=X_t%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BMxD%7D">,
can be easily transformed into their PCA representation, <img src="https://render.githubusercontent.com/render/math?math=%5Ctilde%7BY%7D_t%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BMxH%7D">.

First center the test samples with the mean of the training samples.
<br />
<img src="https://render.githubusercontent.com/render/math?math=%5Ctilde%7BX%7D_t%20%3D%20X_t%20-%201_%7BM%7D%5Chat%7B%5Cmu%7D%5E%7BT%7D_%7Bx%7D">

Transform the centered data using the first H primary components.
<br />
<img src="https://render.githubusercontent.com/render/math?math=%5Ctilde%7BY%7D_t%20%3D%20%5Ctilde%7BX%7D_t%5Cnu_%7BH%7D">

</encoder>



### Decoder
<decoder>

</decoder>

## Results

### 8x8 Bit Digits
<eight>

</eight>

### 28x28 Bit Digits
<twentyeight>

</twentyeight>

### Yale Faces
<faces>

</faces>