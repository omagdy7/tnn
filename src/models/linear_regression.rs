use crate::{
    linear_algebra::{matrix::mult_par, matrix::Matrix},
    models::model::Model,
};
use rand::{thread_rng, Rng};

// dx/dw =  (2 * X^T * (X * W + b - Y)) / Y.rows
fn dcost_w(x: &Matrix, y: &Matrix, w: &Matrix, b: f64) -> Matrix {
    let mut predictions = x * w;
    predictions.add_element_wise(b);
    let errors = &predictions - y;
    let x_t = x.transpose();
    let mut gradient_w = x_t * errors;
    gradient_w.divide_element_wise(y.rows as f64);
    gradient_w
}

// dx/db = 2 * x_i * w + b - y_i
fn dcost_b(x: &Matrix, y: &Matrix, w: &Matrix, b: f64) -> f64 {
    let mut predictions = x * w;
    predictions.add_element_wise(b);
    let errors = &predictions - y;
    errors.elements.iter().sum::<f64>() / y.rows as f64
}

pub struct LinearRegression {
    weights: Matrix,
    bias: f64,
}

impl LinearRegression {
    pub fn new(num_of_features: usize) -> Self {
        let mut rng = thread_rng();
        let mut weights: Vec<f64> = vec![0.0; num_of_features];
        for i in 0..num_of_features {
            weights[i] = rng.gen_range(0.0..3.00);
        }
        let weights = Matrix::with_vector(weights, num_of_features, 1);
        let bias = rng.gen_range(0.0..3.00);
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
    ) -> f64 {
        let xs = &xs.transpose();

        for _ in 0..epochs {
            let db = dcost_b(xs, y, &self.weights, self.bias);
            let mut gradient_w = dcost_w(xs, y, &self.weights, self.bias);
            gradient_w.prod_element_wise(learning_rate);
            self.weights -= gradient_w;
            self.bias -= learning_rate * db;

            // println!(
            //     "MSE = {:?}, weights = {:?}, bias = {}",
            //     self.cost(xs, y, &self.weights, self.bias, &activation_function),
            //     self.weights,
            //     self.bias,
            // );
        }
        println!(
            "MSE = {:?}",
            self.cost(xs, y, &self.weights, self.bias, &activation_function)
        );
        self.cost(xs, y, &self.weights, self.bias, &activation_function)
    }

    fn predict<F: Fn(f64) -> f64>(&self, xs: Vec<f64>, activation_function: F) -> f64 {
        let len = xs.len();
        let xs = Matrix::with_vector(xs, 1, len);
        let result = mult_par(&xs, &self.weights);
        activation_function(result.at(0, 0) + self.bias)
    }

    fn dump(&self) {
        println!("weights: {}\nbias: {}", self.weights, self.bias);
    }
}

mod tests {
    use super::*;
    use crate::utils::{get_training_data, identity};

    const EPS: f64 = 0.00001;

    fn test_helper(
        dataset: &str,
        features: usize,
        rate: f64,
        epochs: usize,
        is_normalize: bool,
    ) -> f64 {
        let data = get_training_data(&format!("train/{}.txt", dataset));
        let (rows, cols) = (data.len(), data[0].len());
        let data =
            Matrix::with_vector(data.into_iter().flatten().collect(), rows, cols).transpose();

        println!("data: {data}");
        let (x, y) = (data.rows_n(data.rows - 1), data.row(data.rows - 1));
        let rows = data.rows;
        let cols = data.cols;
        let mut x = Matrix::with_vector(x, rows - 1, cols).transpose();
        if is_normalize {
            x.normalize();
        }
        let y = Matrix::with_vector(y, cols, 1);
        let mut model = LinearRegression::new(features);
        model.fit(&x.transpose(), &y, rate, epochs, identity)
    }

    #[test]
    fn double_basic() {
        assert!(test_helper("double", 1, 1.25, 1000, true) - 0.0 < EPS)
    }

    #[test]
    fn double_with_bias() {
        assert!(test_helper("double_plus_one", 1, 1.25, 1000, true) - 0.0 < EPS)
    }

    #[test]
    fn add_basic() {
        assert!(test_helper("add", 2, 1.0, 5000, true) - 0.0 < EPS)
    }

    #[test]
    fn add_complex() {
        assert!(test_helper("complex_add", 3, 0.75, 1000 * 10, true) - 0.0 < EPS)
    }

    #[test]
    fn four_times_plus_seven() {
        assert!(test_helper("four_times_plus_seven", 1, 1.5, 2000, true) - 0.0 < EPS)
    }
}
