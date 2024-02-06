use crate::{
    linear_algebra::{matrix::Matrix, par_mult::mult_par},
    model::Model,
    utils::helper,
};
use rand::{thread_rng, Rng};

// dx/dw =  (2 * X^T * (X * W + b - Y)) / Y.rows
fn dcost_w(x: &Matrix, y: &Matrix, w: &Matrix, b: f64) -> Matrix {
    let mut predictions = mult_par(x, w);
    predictions.add_element_wise(b);
    let errors = predictions.sub(y);
    let x_t = x.transpose();
    let mut gradient_w = mult_par(&x_t, &errors);
    gradient_w.divide_element_wise(y.rows as f64);
    gradient_w
}

// dx/db = 2 * x_i * w + b - y_i
fn dcost_b(x: &Matrix, y: &Matrix, w: &Matrix, b: f64) -> f64 {
    let mut predictions = mult_par(x, w);
    predictions.add_element_wise(b);
    let errors = predictions.sub(y);
    errors.elements.iter().sum::<f64>() / y.rows as f64
}

pub struct PolynomialRegression {
    weights: Matrix,
    bias: f64,
    degree: usize,
}

impl PolynomialRegression {
    pub fn new(num_of_features: usize, degree: usize) -> Self {
        let mut rng = thread_rng();
        let weights_len = num_of_features * degree + (num_of_features - 1) * (num_of_features) / 2;
        let mut weights: Vec<f64> = vec![0.0; weights_len];
        for w in weights.iter_mut() {
            *w = rng.gen_range(0.00..3.00);
        }
        let weights = Matrix::with_vector(weights, weights_len, 1);
        let bias = rng.gen_range(0.00..1.00);
        PolynomialRegression {
            degree,
            weights,
            bias,
        }
    }
}

impl Model for PolynomialRegression {
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
        let mut elements = vec![];
        for c in 0..xs.cols {
            let mut x_c = vec![0.0; xs.rows];
            for j in 0..xs.rows {
                x_c[j] = xs.elements[j + c * xs.rows];
            }
            elements.push(x_c);
            for i in 1..self.degree {
                let mut x_degree_i = vec![0.0; xs.rows];
                for j in 0..xs.rows {
                    x_degree_i[j] = xs.elements[j + c * xs.rows].powi(i as i32 + 1);
                }
                elements.push(x_degree_i);
            }
            for cc in c + 1..xs.cols {
                let mut prod = vec![0.0; xs.rows];
                for j in 0..xs.rows {
                    prod[j] = xs.elements[j + c * xs.rows] * xs.elements[j + cc * xs.rows];
                }
                elements.push(prod);
            }
        }

        // println!("xs: {}", xs);

        let elements = helper(&elements);
        let elements: Vec<f64> = elements.into_iter().flatten().collect();

        let xs = &Matrix::with_vector(elements, xs.rows, self.weights.rows);

        println!("xs: {}", xs);

        for _ in 0..epochs {
            let db = dcost_b(xs, y, &self.weights, self.bias);
            let mut gradient_w = dcost_w(xs, y, &self.weights, self.bias);
            gradient_w.prod_element_wise(learning_rate);
            self.weights.sub(&gradient_w);
            self.bias -= learning_rate * db;
        }
        println!(
            "MSE = {:?}",
            self.cost(xs, y, &self.weights, self.bias, activation_function)
        );
    }

    fn predict<F: Fn(f64) -> f64>(&self, xs: Vec<f64>, activation_function: F) -> f64 {
        let mut nxs = vec![];
        for i in 0..xs.len() {
            nxs.push(xs[i]);
            for j in 1..self.degree {
                nxs.push(xs[i].powi(j as i32 + 1))
            }
            for j in i + 1..xs.len() {
                nxs.push(xs[i] * xs[j])
            }
        }
        let len = xs.len();
        let xs = Matrix::with_vector(nxs, 1, self.weights.rows);
        println!("here: {xs}");
        let result = mult_par(&xs, &self.weights);
        activation_function(result.at(0, 0) + self.bias)
    }

    fn dump(&self) {
        println!("weights: {}\nbias: {}", self.weights, self.bias);
    }
}
