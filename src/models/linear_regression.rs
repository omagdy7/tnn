use crate::{
    linear_algebra::{matrix::mult_par, matrix::Matrix},
    models::model::Model,
    utils::helper,
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
        let mut elements = vec![];
        for c in 0..xs.cols {
            let mut x_c = vec![0.0; xs.rows];
            for j in 0..xs.rows {
                x_c[j] = xs.elements[j + c * xs.rows];
            }
            elements.push(x_c);
        }
        let elements = helper(&elements);
        let elements: Vec<f64> = elements.into_iter().flatten().collect();
        let xs = &Matrix::with_vector(elements, xs.rows, self.weights.rows);

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
    use crate::utils::{get_training_data, identity, normalize};

    const EPS: f64 = 0.00001;

    fn test_helper(
        dataset: &str,
        features: usize,
        rate: f64,
        epochs: usize,
        is_normalize: bool,
    ) -> f64 {
        let mut model = LinearRegression::new(features);
        let data = get_training_data(&format!("train/{}.txt", dataset));
        if is_normalize {
            let (data, _, _) = normalize(data);
            let (x, y) = (
                data[0..data.len() - 1].to_vec(),
                data.last().unwrap().to_vec(),
            );
            let cols = x[0].len();
            let rows = x.len();
            let final_x: Vec<f64> = x.into_iter().flatten().collect();
            let x = Matrix::with_vector(final_x, cols, rows);
            let y = Matrix::with_vector(y, cols, 1);
            let error = model.fit(&x, &y, rate, epochs, identity);
            error
        } else {
            let (x, y) = (
                data[0..data.len() - 1].to_vec(),
                data.last().unwrap().to_vec(),
            );

            let cols = x[0].len();
            let rows = x.len();
            let final_x: Vec<f64> = x.into_iter().flatten().collect();
            let x = Matrix::with_vector(final_x, cols, rows);
            let y = Matrix::with_vector(y, cols, 1);
            let error = model.fit(&x, &y, rate, epochs, identity);
            error
        }
    }

    #[test]
    fn double_basic() {
        assert!(test_helper("double", 1, 0.5, 1000, true) - 0.0 < EPS)
    }

    #[test]
    fn double_with_bias() {
        assert!(test_helper("double_plus_one", 1, 0.5, 1000, true) - 0.0 < EPS)
    }

    #[test]
    fn add_basic() {
        assert!(test_helper("add", 2, 1.0, 1000, true) - 0.0 < EPS)
    }

    #[test]
    fn add_complex() {
        assert!(test_helper("complex_add", 3, 1.0, 10000, true) - 0.0 < EPS)
    }

    #[test]
    fn four_times_plus_seven() {
        assert!(test_helper("four_times_plus_seven", 1, 1.0, 1000, true) - 0.0 < EPS)
    }
}
