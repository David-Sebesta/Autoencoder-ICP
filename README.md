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
The encoding function, <img src="https://render.githubusercontent.com/render/math?math=f%20%3A%20%5Cmathbb%7BR%7D%5E%7BD%7D%20%5Cto%20%5Cmathbb%7BR%7D%5E%7BH%7D">,
maps D-dimensions to H-dimensions.
The decoding function, <img src="https://render.githubusercontent.com/render/math?math=g%20%3A%20%5Cmathbb%7BR%7D%5E%7BH%7D%20%5Cto%20%5Cmathbb%7BR%7D%5E%7BD%7D">,
maps H-dimensions to D-dimensions.
The reconstructed samples, <img src="https://render.githubusercontent.com/render/math?math=%5Chat%7Bx%7D%20%5Ctriangleq%20g(f(x))%20%5Capprox%20x">,
after being encoded then decoded, are approximately equal to the original samples.
</technique>

### Encoder
<encoder>
The encoder first uses PCA to find the transformation. With training samples <img src="https://render.githubusercontent.com/render/math?math=X%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BRxD%7D">
and total variance explained <img src="https://render.githubusercontent.com/render/math?math=0%20%5Cle%20p%20%5Cle%201">, the
PCA training is as follows.



</encoder>

h<sub>&theta;</sub>(x) = &theta;<sub>o</sub> x + &theta;<sub>1</sub>x

![formula](https://render.githubusercontent.com/render/math?math=A%20-%20B%20=%20\{x%20\in%20\U%20\mid%20x%20\in%20A%20\land%20x%20\notin%20B%20\})

\mathbb{R}



<img src="https://render.githubusercontent.com/render/math?math=e^{i %2B\pi} =x%2B1">


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