use crate::linear_algebra::matrix::Matrix;

pub trait Model {
    fn cost<F: Fn(f64) -> f64>(
        &self,
        xs: &Matrix,
        y: &Matrix,
        w: &Matrix,
        b: f64,
        activation_function: F,
    ) -> f64;
    fn fit<F: Fn(f64) -> f64>(
        &mut self,
        xs: &Matrix,
        y: &Matrix,
        learning_rate: f64,
        epochs: usize,
        activation_function: F,
    ) -> f64;
    fn predict<F: Fn(f64) -> f64>(&self, xs: Vec<f64>, activation_function: F) -> f64;
    fn dump(&self);
}
