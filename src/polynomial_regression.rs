use crate::{
    linear_algebra::{matrix::Matrix, par_mult::mult_par},
    model::Model,
    utils::helper,
};
use rand::{thread_rng, Rng};

pub struct PolynomialRegression {
    degree: usize,
    weights: Matrix,
    bias: f64,
}

impl PolynomialRegression {
    pub fn new(num_of_features: usize, degree: usize) -> Self {
        let mut rng = thread_rng();
        let mut weights: Vec<f64> = vec![0.0; num_of_features * degree];
        for w in weights.iter_mut() {
            *w = rng.gen_range(0.00..3.00);
        }
        let weights = Matrix::with_vector(weights, num_of_features * degree, 1);
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
        }

        // println!("xs: {}", xs);

        let elements = helper(&elements);
        let elements: Vec<f64> = elements.into_iter().flatten().collect();
        let xs = &Matrix::with_vector(elements, xs.rows, self.weights.rows);

        println!("xs: {}", xs);

        let eps = 0.05;
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
        let mut nxs = vec![];
        for i in 0..xs.len() {
            nxs.push(xs[i]);
            for j in 1..self.degree {
                nxs.push(xs[i].powi(j as i32 + 1))
            }
        }
        let len = xs.len();
        let xs = Matrix::with_vector(nxs, 1, len * self.degree);
        println!("{xs}");
        let result = mult_par(&xs, &self.weights);
        activation_function(result.at(0, 0) + self.bias)
    }

    fn dump(&self) {
        println!("weights: {}\nbias: {}", self.weights, self.bias);
    }
}
