use crate::{
    linear_algebra::{
        matrix::Matrix,
        mult::mult_transpose,
        par_mult::{self, mult_par, mult_par_strassen},
    },
    model::Model,
};
use rand::{thread_rng, Rng};

pub struct LinearRegression {
    weights: Matrix,
    bias: f64,
}

impl LinearRegression {
    pub fn new(num_of_features: usize) -> Self {
        let mut rng = thread_rng();
        let mut weights: Vec<f64> = vec![0.0; num_of_features];
        for i in 0..num_of_features {
            weights[i] = rng.gen_range(0.00..3.00);
        }
        let weights = Matrix::with_vector(weights, num_of_features, 1);
        let bias = rng.gen_range(0.00..1.00);
        LinearRegression { weights, bias }
    }
}

impl Model for LinearRegression {
    fn cost<F: Fn(f64) -> f64>(
        &self,
        xs: &Matrix,
        y: &Matrix,
        w: &Matrix,
        b: f64,
        activation_function: F,
    ) -> f64 {
        let mut mse = 0.0;
        let predicted_values = mult_par(xs, w);
        for (predicted, actual) in predicted_values.elements.iter().zip(y.elements.iter()) {
            mse += f64::powi(actual - activation_function(predicted + b), 2);
        }
        mse /= 2.0 * y.rows as f64;
        mse
    }

    fn fit<F: Fn(f64) -> f64>(
        &mut self,
        xs: &Matrix,
        y: &Matrix,
        learning_rate: f64,
        epochs: usize,
        activation_function: F,
    ) {
        let eps = 0.001;
        for _ in 0..epochs {
            let c = self.cost(xs, y, &self.weights, self.bias, &activation_function);
            let db =
                (self.cost(xs, y, &self.weights, self.bias + eps, &activation_function) - c) / eps;
            for i in 0..self.weights.rows {
                let mut tmp_weights = self.weights.clone();
                tmp_weights.elements[i] += eps;
                let cost_i = self.cost(xs, y, &tmp_weights, self.bias, &activation_function);
                let dw = (cost_i - c) / eps;
                self.weights.elements[i] -= learning_rate * dw;
            }
            self.bias -= learning_rate * db;
            // println!(
            //     "MSE = {:?}, self.weights = {:?}",
            //     self.cost(xs, y, &self.weights, self.bias, &activation_function),
            //     self.weights,
            // );
        }
        println!(
            "MSE = {:?}",
            self.cost(xs, y, &self.weights, self.bias, activation_function)
        );
    }

    fn predict<F: Fn(f64) -> f64>(&self, xs: Vec<f64>, activation_function: F) -> f64 {
        let len = xs.len();
        let xs = Matrix::with_vector(xs, 1, len);
        let result = mult_par(&xs, &self.weights);
        activation_function(result.at(0, 0) + self.bias)
    }

    fn dump(&self) {
        println!("weights: {:?}\nbias: {}", self.weights, self.bias);
    }
}
