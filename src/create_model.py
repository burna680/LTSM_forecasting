import tensorflow as tf
from keras import layers

class LSTMForecasting(tf.keras.Model):
    def __init__(self, input_size, lstm_hidden_size, linear_hidden_size, 
                 lstm_num_layers, linear_num_layers, output_size, **kwargs):
        super(LSTMForecasting, self).__init__(**kwargs)

        # Store parameters for get_config
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.linear_hidden_size = linear_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.linear_num_layers = linear_num_layers
        self.output_size = output_size

        # Stack multiple LSTM layers if lstm_num_layers > 1
        self.lstm_layers = []
        for i in range(lstm_num_layers):
            return_sequences = (i < lstm_num_layers - 1)  # Only the last LSTM layer should not return sequences
            self.lstm_layers.append(
                layers.LSTM(lstm_hidden_size, return_sequences=return_sequences, 
                            recurrent_initializer='glorot_uniform', 
                            stateful=False)
            )
        
        # Linear layers
        self.linear_layers = []
        self.linear_layers.append(layers.Dense(linear_hidden_size, activation='relu'))

        for _ in range(linear_num_layers - 1):
            linear_hidden_size = int(linear_hidden_size / 1.5)
            self.linear_layers.append(layers.Dense(linear_hidden_size, activation='relu'))
        
        # Final output layer
        self.fc = layers.Dense(output_size)

    def call(self, x, training=False):
        # Ensure input is 3D (batch_size, timesteps, features)
        if len(x.shape) < 3:
            x = tf.expand_dims(x, axis=1)  # Add a timestep dimension if needed
        
        # Passing through stacked LSTM layers
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x)
        
        # Passing through each dense layer
        for layer in self.linear_layers:
            x = layer(x)
        
        # Final dense layer (only take the last output from LSTM if 3D)
        if len(x.shape) == 3:
            x = self.fc(x[:, -1, :])  # Take the last timestep output
        else:
            x = self.fc(x)  # If not 3D, pass as is
        
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_size': self.input_size,
            'lstm_hidden_size': self.lstm_hidden_size,
            'linear_hidden_size': self.linear_hidden_size,
            'lstm_num_layers': self.lstm_num_layers,
            'linear_num_layers': self.linear_num_layers,
            'output_size': self.output_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
