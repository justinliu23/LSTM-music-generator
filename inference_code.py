def inference_model(LSTM_cell, densor, n_x = 90, n_a = 64, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_x -- number of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_x))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    outputs = []
    
    for t in range(Ty):
        
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
        x = RepeatVector(1)(out)
        
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    
    return inference_model


inference_model = inference_model(LSTM_cell, densor)

x1 = np.zeros((1, 1, 90))
x1[:,:,35] = 1
a1 = np.zeros((1, n_a))
c1 = np.zeros((1, n_a))
predicting = inference_model.predict([x1, a1, c1])

indices = np.argmax(predicting, axis = -1)
results = to_categorical(indices, num_classes=90)
