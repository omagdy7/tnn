struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    fn new(weights: Vec<f64>, bias: f64) -> Self {
        Neuron { weights, bias }
    }
}

struct Layer<A: Fn(f64) -> f64> {
    neurons: Vec<Neuron>,
    activation_function: A,
}

impl<A> Layer<A>
where
    A: Fn(f64) -> f64,
{
    fn new(neurons: Vec<Neuron>, activation_function: A) -> Self {
        Layer {
            neurons,
            activation_function,
        }
    }
}

struct NeuralNetwork<A: Fn(f64) -> f64> {
    layers: Vec<Layer<A>>,
}

impl<A> NeuralNetwork<A>
where
    A: Fn(f64) -> f64,
{
    fn new(layers: Vec<Layer<A>>) -> Self {
        NeuralNetwork { layers }
    }
}
