class FeedForwardNeuralNetwork:
    def __init__(self, num_features, num_hidden_neurons, num_output_neurons):
        self.num_features = num_features
        self.num_hidden_neurons = num_hidden_neurons
        self.num_output_neurons = num_output_neurons
        self.hidden_weights, self.hidden_bias, self.output_weights, self.output_bias = self.initialize_weights()

    def initialize_weights(self):
        # Hidden Unit Weight & Bias Initialization
        hidden_weights = np.random.randn(self.num_features, self.num_hidden_neurons)
        hidden_bias = np.random.randn(1, self.num_hidden_neurons)

        # Output Layer Weight & Bias Initialization
        output_weights = np.random.randn(self.num_hidden_neurons, self.num_output_neurons)
        output_bias = np.random.randn(1, self.num_output_neurons)

        return hidden_weights, hidden_bias, output_weights, output_bias

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def rmse(y_true, y_pred):
        squared_error = np.square(np.subtract(y_true, y_pred))
        mean_squared_error = np.mean(squared_error)
        rmse = np.sqrt(mean_squared_error)
        return rmse

    @staticmethod
    def mse(y_true, y_pred):
        squared_error = np.square(np.subtract(y_true, y_pred))
        mean_squared_error = np.mean(squared_error)
        return mean_squared_error

    @staticmethod
    def mae(y_true, y_pred):
        absolute_error = np.abs(np.subtract(y_true, y_pred))
        mean_absolute_error = np.mean(absolute_error)
        return mean_absolute_error

    def forward_pass(self, input_data, activation_function, loss_function):
        # Check if dimensions are compatible
        if input_data.shape[1] != self.hidden_weights.shape[0] or self.hidden_weights.shape[1] != self.output_weights.shape[0]:
            raise ValueError("Input data and weight dimensions are not compatible.")

        # Compute the input to the hidden layer
        hidden_input = np.dot(input_data, self.hidden_weights) + self.hidden_bias

        # Apply the activation function to the hidden layer
        hidden_output = activation_function(hidden_input)

        # Compute the input to the output layer
        output_input = np.dot(hidden_output, self.output_weights) + self.output_bias

        # Calculate the error using the provided loss function
        error = loss_function(input_data, output_input)

        # Return both error and predicted outputs
        return error, output_input
