# LSTM-music-generator

Click into ```Improvise_a_Jazz_Solo_with_an_LSTM_Network.ipynb``` to listen to a 30-second audio clip generated using this algorithm!

Here is the architecture of the model I implemented:

![image](https://github.com/user-attachments/assets/2bc6b993-da45-47ae-889d-3ee77b54b6f4)

* $X = (x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, \cdots, x^{\langle T_x \rangle})$ is a window of size $T_x$ scanned over the musical corpus. 
* Each $x^{\langle t \rangle}$ is an index corresponding to a value.
* $\hat{y}^{\langle t \rangle}$ is the prediction for the next value.
* I trained the model on random snippets of 30 values taken from a much longer piece of music.

![image](https://github.com/user-attachments/assets/57c1a794-eb4c-46dc-854e-3ba1be91264f)

At each step of sampling, I:
* Take as input the activation '`a`' and cell state '`c`' from the previous state of the LSTM.
* Forward propagate by one step.
* Get a new output activation, as well as cell state. 
* The new activation '`a`' can then be used to generate the output using the fully connected layer, `densor`. 

The ideas presented in this notebook came primarily from three computational music papers cited below. The implementation here also took significant inspiration and used many components from Ji-Sung Kim's GitHub repository.

- Ji-Sung Kim, 2016, [deepjazz](https://github.com/jisungk/deepjazz)
- Jon Gillick, Kevin Tang and Robert Keller, 2009. [Learning Jazz Grammars](http://ai.stanford.edu/~kdtang/papers/smc09-jazzgrammar.pdf)
- Robert Keller and David Morrison, 2007, [A Grammatical Approach to Automatic Improvisation](http://smc07.uoa.gr/SMC07%20Proceedings/SMC07%20Paper%2055.pdf)
- François Pachet, 1999, [Surprising Harmonies](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.5.7473&rep=rep1&type=pdf)
