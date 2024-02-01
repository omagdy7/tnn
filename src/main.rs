mod linear_regression;
mod model;
mod polynomial_regression;
mod utils;
use linear_regression::*;
use model::Model;
use utils::{get_training_data, identity};

use crate::{
    polynomial_regression::PolynomialRegression,
    utils::{normalize, sigmoid},
};

fn main() {
    // let mut linear_reg = LinearRegression::new(1);
    // linear_reg.fit("train/double.txt", 0.0001, 1000 * 1000 * 10, identity);
    // linear_reg.dump();
    // println!("{}", linear_reg.predict(vec![40.0], identity));
    //
    // let mut model = LinearRegression::new(2);
    // model.fit("train/or_gate.txt", 0.01, 100_000_00, sigmoid);
    // println!("0 | 0 = {}", model.predict(vec![0.0, 0.0], sigmoid));
    // println!("0 | 1 = {}", model.predict(vec![0.0, 1.0], sigmoid));
    // println!("1 | 0 = {}", model.predict(vec![1.0, 0.0], sigmoid));
    // println!("1 | 1 = {}", model.predict(vec![1.0, 1.0], sigmoid));
    // model.dump();

    let data = get_training_data("train/complex_polynomial.txt");
    let (mut normalized_data, min_x, max_x) = normalize(data);

    let mut model = PolynomialRegression::new(3);
    model.fit(&mut normalized_data, 0.004, 100_000_000, identity);
    let x = (50.0 - min_x) / (max_x - min_x);
    println!("prediction: {}", model.predict(vec![x], identity));
    model.dump();
}
