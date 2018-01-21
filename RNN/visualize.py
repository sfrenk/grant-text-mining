# Heavily inspired by https://medium.com/@plusepsilon/visualizations-of-recurrent-neural-networks-c18f07779d56

from keras.models import Sequential, load_model
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence

# Load model
model = load_model('model.h5')

# TO DO: infer these from model rather than hard-coding
vocab_size = 50229
maxlen = 80

# Visualization function from the Medium post

def visualize_model(model, include_gradients=False):
    recurrent_layer = model.get_layer('recurrent_layer')
    output_layer = model.get_layer('output_layer')

    inputs = []
    inputs.extend(model.inputs)

    outputs = []
    outputs.extend(model.outputs)
    outputs.append(recurrent_layer.output)
    outputs.append(recurrent_layer.W_f)  # -- weights of the forget gates (assuming LSTM)

    if include_gradients:
        loss = K.mean(model.output)  # [batch_size, 1] -> scalar
        grads = K.gradients(loss, recurrent_layer.output)
        grads_norm = grads / (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        outputs.append(grads_norm)

    all_function = K.function(inputs, outputs)
    output_function = K.function([output_layer.input], model.outputs)
    return all_function, output_function

# -- predict
all_function, output_function = visualize_model(model, include_gradients=True)

# Get example
with open('myfile.txt', 'r') as f:
    example = f.readline()

# Process example
X = one_hot(example, vocab_size) 
X = sequence.pad_sequences(test_abstracts, maxlen=maxlen)

# -- Return scores, raw rnn values and gradients
# scores is equivalent to model.predict(X)
scores, rnn_values, rnn_gradients, W_i = all_function([X])
print(scores.shape, rnn_values.shape, rnn_gradients.shape, W_i.shape)

# -- score prediction
print("Scores:", scores)

# -- Return scores at each step in the time sequence
time_distributed_scores = map(lambda x: output_function([x]), rnn_values)
print("Time distributed (word-level) scores:", map(lambda x: x[0], time_distributed_scores))