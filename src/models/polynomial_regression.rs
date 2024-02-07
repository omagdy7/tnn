use crate::{
    linear_algebra::{matrix::mult_par, matrix::Matrix},
    models::model::Model,
    utils::helper,
};
use rand::{thread_rng, Rng};

fn prepare_data(xs: &Matrix, degree: usize, rows: usize, include_interactions: bool) -> Matrix {
    let mut elements = vec![];
    for c in 0..xs.cols {
        let mut x_c = vec![0.0; xs.rows];
        for j in 0..xs.rows {
            x_c[j] = xs.elements[j + c * xs.rows];
        }
        elements.push(x_c);
        for i in 1..degree {
            let mut x_degree_i = vec![0.0; xs.rows];
            for j in 0..xs.rows {
                x_degree_i[j] = xs.elements[j + c * xs.rows].powi(i as i32 + 1);
            }
            elements.push(x_degree_i);
        }
        if include_interactions {
            for cc in c + 1..xs.cols {
                let mut prod = vec![0.0; xs.rows];
                for j in 0..xs.rows {
                    prod[j] = xs.elements[j + c * xs.rows] * xs.elements[j + cc * xs.rows];
                }
                elements.push(prod);
            }
        }
    }
    println!("{xs}");
    let elements = helper(&elements);
    let elements: Vec<f64> = elements.into_iter().flatten().collect();
    let xs = Matrix::with_vector(elements, xs.rows, rows);
    println!("{xs}");
    xs
}

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

pub struct PolynomialRegression {
    weights: Matrix,
    bias: f64,
    degree: usize,
    include_interactions: bool,
}

impl PolynomialRegression {
    pub fn new(num_of_features: usize, degree: usize, include_interactions: bool) -> Self {
        let mut rng = thread_rng();
        let mut weights_len = num_of_features * degree;
        if include_interactions {
            weights_len = num_of_features * degree + (num_of_features - 1) * (num_of_features) / 2;
        }
        let mut weights: Vec<f64> = vec![0.0; weights_len];
        for w in weights.iter_mut() {
            *w = rng.gen_range(0.00..100.00);
        }
        let weights = Matrix::with_vector(weights, weights_len, 1);
        let bias = rng.gen_range(0.00..100.00);
        PolynomialRegression {
            degree,
            weights,
            bias,
            include_interactions,
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
    ) -> f64 {
        let xs = &prepare_data(
            xs,
            self.degree,
            self.weights.rows,
            self.include_interactions,
        );

        for _ in 0..epochs {
            let db = dcost_b(xs, y, &self.weights, self.bias);
            let mut gradient_w = dcost_w(xs, y, &self.weights, self.bias);
            gradient_w.prod_element_wise(learning_rate);
            self.weights -= gradient_w;
            self.bias -= learning_rate * db;
        }
        println!(
            "MSE = {:?}",
            self.cost(xs, y, &self.weights, self.bias, &activation_function)
        );
        self.cost(xs, y, &self.weights, self.bias, &activation_function)
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
        let xs = Matrix::with_vector(nxs, 1, self.weights.rows);
        let result = &xs * &self.weights;
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
        degree: usize,
        include_interactions: bool,
        rate: f64,
        epochs: usize,
        is_normalize: bool,
    ) -> f64 {
        let mut model = PolynomialRegression::new(features, degree, include_interactions);
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
    fn square() {
        assert!(test_helper("square", 1, 2, false, 1.25, 2000, true) - 0.0 < EPS)
    }

    #[test]
    fn cube() {
        assert!(test_helper("cube", 1, 3, false, 1.25, 5000 * 10, true) - 0.0 < EPS)
    }

    #[test]
    fn product() {
        assert!(test_helper("product", 2, 1, true, 1.25, 1000 * 10, true) - 0.0 < EPS)
    }

    #[test]
    fn two_features_square() {
        assert!(test_helper("polynomial_and_polynomial", 2, 2, true, 0.5, 5000, true) - 0.0 < EPS)
    }

    #[test]
    fn complex_polynomial() {
        assert!(
            test_helper(
                "complex_polynomial",
                3,
                3,
                false,
                1.14,
                1000 * 1000 * 100,
                true
            ) - 0.0
                < EPS
        )
    }
}
