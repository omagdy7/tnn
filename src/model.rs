pub trait Model {
    fn cost<F: Fn(f64) -> f64>(
        &self,
        training_data: &Vec<Vec<f64>>,
        w: &[f64],
        b: f64,
        activation_function: F,
    ) -> f64;
    fn fit<F: Fn(f64) -> f64>(
        &mut self,
        training_data: &mut Vec<Vec<f64>>,
        learning_rate: f64,
        epochs: usize,
        activation_function: F,
    );
    fn predict<F: Fn(f64) -> f64>(&self, xs: Vec<f64>, activation_function: F) -> f64;
    fn dump(&self);
}
