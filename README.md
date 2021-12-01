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
<img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7Bf%3A%5Cmathbb%7BR%7D%5E%7BD%7D%5Cto%5Cmathbb%7BR%7D%5E%7BH%7D%7D">

The decoding function maps H-dimensions to D-dimensions.
<img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7Bg%20%3A%20%5Cmathbb%7BR%7D%5E%7BH%7D%20%5Cto%20%5Cmathbb%7BR%7D%5E%7BD%7D%7D">

The reconstructed samples after being encoded then decoded, are approximately equal to the original samples.
<img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7B%5Chat%7Bx%7D%20%5Ctriangleq%20g(f(x))%20%5Capprox%20x%7D">
</technique>

### Encoder
<encoder>
The encoder first uses PCA based on the sample covariance matrix to find the transformation.
With training samples <img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7BX%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BRxD%7D%7D">
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

The decoded test samples are calculated using the same H principle components, and then uncentering samples with the training samples mean.
<br />
<img src="https://render.githubusercontent.com/render/math?math=%5Cbbox%5Bwhite%5D%7B%5Chat%7BX%7D_%7Bt%7D%3D%5Ctilde%7BY%7D_%7Bt%7D%5Cnu%5E%7BT%7D_%7BH%7D%20%2B%5Chat%7B%5Cmu%7D_%7Bx%7D%7D">

</decoder>

## Results
<results>
For the data used in this project, N is the number of samples and D is the number of pixels in that sample.
Each sample is a D-length vector.
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

<br />
With the total variance explained, p, equal to 0.85. The total number of principle components is 17.

![plot](results/8x8/one_p_0.85.png?raw=true )

![plot](results/8x8/three_p_0.85.png?raw=true )

<br />
With the total variance explained, p, equal to 0.9. The total number of principle components is 21.

![plot](results/8x8/two_p_0.9.png?raw=true )

![plot](results/8x8/three_p_0.9.png?raw=true )

<br />
With the total variance explained, p, equal to 0.95. The total number of principle components is 29.

![plot](results/8x8/six_p_0.95.png?raw=true )

![plot](results/8x8/one_p_0.95.png?raw=true )

<br />
With the total variance explained, p, equal to 0.99. The total number of principle components is 41.

![plot](results/8x8/five_p_0.99.png?raw=true )

![plot](results/8x8/two_p_0.99.png?raw=true )

#### Noise
Now lets add some uniform noise to the images.

With the total variance explained, p, equal to 0.9.

![plot](results/8x8/noise_zero_p_0.9.png?raw=true )

<br />
With the total variance explained, p, equal to 0.95.

![plot](results/8x8/noise_one_p_0.95.png?raw=true )

The images are somewhat denoised, but lets take it up a notch when we get to the 28x28 digits.

#### Cumulative Total Variance Explained
This graph displays the Cumulative Total Variance Explained vs the Number of Components.
See that 90% is at N=21.
Since the other data has a lot more dimensions, the graph gets way to busy to read, but the first couple components
always contain the majority of the total variance.

![plot](results/8x8/cve.png?raw=true "Cumulative Variance Explained vs Number of Principle Components")
</eight>

### 28x28 Bit Digits
<twentyeight>
Lets take a look at digits with a lot more dimensions. 

With the total variance explained, p, equal to 0.8. The total number of principle components is 44.

![plot](results/28x28/six_p_0.8.png?raw=true )

<br />
With the total variance explained, p, equal to 0.85. The total number of principle components is 59.

![plot](results/28x28/eight_p_0.85.png?raw=true )

<br />
With the total variance explained, p, equal to 0.9. The total number of principle components is 87.

![plot](results/28x28/four_p_0.9.png?raw=true )

<br />
With the total variance explained, p, equal to 0.95. The total number of principle components is 154.

![plot](results/28x28/four_p_0.95.png?raw=true )

![plot](results/28x28/seven_p_0.95.png?raw=true )

<br />
With the total variance explained, p, equal to 0.99. The total number of principle components is 331.

![plot](results/28x28/zero_p_0.99.png?raw=true )

![plot](results/28x28/four_p_0.99.png?raw=true )

#### Noise

Again, lets add some noise and see what happens.

With the total variance explained, p, equal to 0.8.

![plot](results/28x28/noise_zero_p_0.8.png?raw=true )

<br />
With the total variance explained, p, equal to 0.9.

![plot](results/28x28/noise_one_p_0.9.png?raw=true )

![plot](results/28x28/noise_seven_p_0.9.png?raw=true )

The autoencoder seems to be able to denoise the images. Although not perfect, since this is a simple linear enocoder,
it still does a good job.

</twentyeight>

### Yale Faces
<faces>

The faces have been downsampled by 4, so they are 50x50 with D = 2500.
Since all the faces are not centered, the PCA gets slightly off due to the position of the faces.

With the total variance explained, p, equal to 0.8. The total number of principle components is 11.

![plot](results/faces/p_0.8_1.png?raw=true)

<br />
With the total variance explained, p, equal to 0.85. The total number of principle components is 17.

![plot](results/faces/p_0.85_1.png?raw=true)

![plot](results/faces/p_0.85_2.png?raw=true)

<br />
With the total variance explained, p, equal to 0.9. The total number of principle components is 26.

![plot](results/faces/p_0.9_1.png?raw=true)

![plot](results/faces/p_0.9_2.png?raw=true)

![plot](results/faces/p_0.9_3.png?raw=true)

<br />
With the total variance explained, p, equal to 0.95. The total number of principle components is 50.

![plot](results/faces/p_0.95_1.png?raw=true)

![plot](results/faces/p_0.95_2.png?raw=true)

![plot](results/faces/p_0.95_3.png?raw=true)


<br />
With the total variance explained, p, equal to 0.99. The total number of principle components is 106.
Using only 106 out of the 2500 principle components, the faces look very similar to the original sample.

![plot](results/faces/p_0.99_1.png?raw=true)

![plot](results/faces/p_0.99_2.png?raw=true)

![plot](results/faces/p_0.99_3.png?raw=true)

![plot](results/faces/p_0.99_4.png?raw=true)


</faces>

## How to use
<p>

The autoencoder is defined as a class, so an autoencoder object can be created, trained, and used.
```python
my_auto = autoencoder.autoencoder()  # Create an autoencoder

my_auto.pca_train(samples=my_samples, p=my_total_variance_explained, h=None)  # Train the PCA model.
#  A specific value of h can be given if a certain number of principle components want to be used.

my_auto.encode(test_samples=my_test_samples)  # encode the test samples

my_auto.decode()  # decode the transformed samples
```

<br />
The decoded data is a variable of the autoencoder object and can be accessed.

```python
my_auto.decoded_data  # decoded data is an NxD array

plt.imshow(my_auto.decoded_data[0].reshape(image_shape))  # Reshape the dimensions back to the image shape's
```  
<br />
There are three functions built in to train and display n images for each data set.

```python
#  Train the 8x8 digits
my_auto.train_8x8_digits(p=0.9, h=None, noise_strength=0, n_images=2, plot_cve=False)

#  Train the 28x28 digits
my_auto.train_28x28_digits(p=0.9, h=None, noise_strength=0, n_images=2, plot_cve=False)

#  Train the Yale Faces
my_auto.train_faces(p=0.9, h=None, noise_strength=0, n_images=2, plot_cve=False, downsample=4)
```

</p>