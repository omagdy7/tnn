use crate::model::Model;
use rand::{thread_rng, Rng};

pub struct PolynomialRegression {
    weights: Vec<f64>,
    bias: f64,
}

impl PolynomialRegression {
    pub fn new(degree: usize) -> Self {
        let mut rng = thread_rng();
        let mut weights: Vec<f64> = vec![0.0; degree];
        for w in weights.iter_mut() {
            *w = rng.gen_range(0.00..3.00);
        }
        let bias = rng.gen_range(0.00..1.00);
        PolynomialRegression { weights, bias }
    }
}

impl Model for PolynomialRegression {
    fn cost<F: Fn(f64) -> f64>(
        &self,
        training_data: &Vec<Vec<f64>>,
        w: &[f64],
        b: f64,
        activation_function: F,
    ) -> f64 {
        let mut mse = 0.0;
        for data in training_data.iter() {
            let (xs, y) = (&data[0..data.len() - 1], data.last().unwrap());
            let mut predicted_result = b;
            for (i, x) in xs.iter().enumerate() {
                predicted_result += x * w[i];
            }
            predicted_result = activation_function(predicted_result);
            // println!("predicted = {predicted_result}, actual: {y}");
            let actual_result = y;
            mse += f64::powi(actual_result - predicted_result, 2);
        }
        mse /= 2.0 * training_data.len() as f64;
        mse
    }

    fn fit<F: Fn(f64) -> f64>(
        &mut self,
        training_data: &mut Vec<Vec<f64>>,
        learning_rate: f64,
        epochs: usize,
        activation_function: F,
    ) {
        for i in 1..self.weights.len() {
            for j in 0..training_data.len() {
                let x = training_data[j][0];
                let y = training_data[j].pop().unwrap();
                training_data[j].push(x.powi(i as i32 + 1));
                training_data[j].push(y);
            }
        }
        // let normalized_data = normalize(normalized_data);
        println!("normalized: {:#?}", training_data);
        let eps = 0.001;
        for _ in 0..epochs {
            let c = self.cost(
                &training_data,
                &self.weights,
                self.bias,
                &activation_function,
            );
            let db = (self.cost(
                &training_data,
                &self.weights,
                self.bias + eps,
                &activation_function,
            ) - c)
                / eps;
            for (i, _) in self.weights.clone().iter().enumerate() {
                let mut tmp_weights = self.weights.clone();
                tmp_weights[i] += eps;
                let cost_i = self.cost(
                    &training_data,
                    &tmp_weights,
                    self.bias,
                    &activation_function,
                );
                let dw = (cost_i - c) / eps;
                self.weights[i] -= learning_rate * dw;
            }
            self.bias -= learning_rate * db;
            // println!(
            //     "MSE = {:?}, self.weights = {:?}",
            //     self.cost(
            //         &normalized_data,
            //         &self.weights,
            //         self.bias,
            //         &activation_function
            //     ),
            //     self.weights,
            // );
        }
        println!(
            "MSE = {:?}",
            self.cost(
                &training_data,
                &self.weights,
                self.bias,
                activation_function
            )
        );
    }

    fn predict<F: Fn(f64) -> f64>(&self, xs: Vec<f64>, activation_function: F) -> f64 {
        let mut result = self.bias;
        for i in 0..self.weights.len() {
            result += xs[0].powi(i as i32 + 1) * self.weights[i]
        }
        activation_function(result)
    }

    fn dump(&self) {
        println!("weights: {:?}\nbias: {}", self.weights, self.bias);
    }
}
