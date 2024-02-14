use tnn::{
    linear_algebra::matrix::Matrix,
    models::linear_regression::*,
    models::model::Model,
    models::polynomial_regression::PolynomialRegression,
    utils::{get_training_data, identity},
};

fn main() {
    // {
    //     let mut linear_reg = LinearRegression::new(3);
    //     let data = get_training_data("train/complex_add.txt");
    //     // let (data, min_x, max_x) = normalize(data);
    //     let (x, y) = (
    //         data[0..data.len() - 1].to_vec(),
    //         data.last().unwrap().to_vec(),
    //     );
    //     let cols = x[0].len();
    //     let rows = x.len();
    //     let final_x: Vec<f64> = x.into_iter().flatten().collect();
    //     let x = Matrix::with_vector(final_x, cols, rows);
    //     let y = Matrix::with_vector(y, cols, 1);
    //
    //     // println!("{}", x);
    //     // println!("{}", y);
    //     linear_reg.fit(&x, &y, 0.0002, 500000, identity);
    //     linear_reg.dump();
    //     println!(
    //         "22 * 2 = {} (should be 44)",
    //         linear_reg.predict(vec![5.0, 63.0, 53.0], identity)
    //     );
    // }

    let data = get_training_data("train/square.txt");
    let (rows, cols) = (data.len(), data[0].len());
    let data = Matrix::with_vector(data.into_iter().flatten().collect(), rows, cols).transpose();

    let (x, y) = (data.rows_n(data.rows - 1), data.row(data.rows - 1));
    let rows = data.rows;
    let cols = data.cols;
    let mut x = Matrix::with_vector(x, rows - 1, cols).transpose();
    x.normalize();
    println!("{x}");
    let y = Matrix::with_vector(y, cols, 1);
    let mut model = PolynomialRegression::new(1, 2, false);
    model.fit(&x.transpose(), &y, 0.5, 5000, identity);
    model.dump();
    println!("{}", model.predict(vec![(24.0)], identity));
}
