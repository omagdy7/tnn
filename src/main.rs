use tnn::{
    linear_algebra::par_mult::Matrix,
    linear_regression::*,
    model::Model,
    polynomial_regression::PolynomialRegression,
    utils::{get_training_data, identity, normalize},
};

fn main() {
    {
        let mut linear_reg = LinearRegression::new(3);
        let data = get_training_data("train/complex_add.txt");
        // let (data, min_x, max_x) = normalize(data);
        let (x, y) = (
            data[0..data.len() - 1].to_vec(),
            data.last().unwrap().to_vec(),
        );
        let cols = x[0].len();
        let rows = x.len();
        let final_x: Vec<f64> = x.into_iter().flatten().collect();
        let x = Matrix::with_vector(final_x, cols, rows);
        let y = Matrix::with_vector(y, cols, 1);

        // println!("{}", x);
        // println!("{}", y);
        linear_reg.fit(&x, &y, 0.0002, 3000000, identity);
        linear_reg.dump();
        println!(
            "22 * 2 = {} (should be 44)",
            linear_reg.predict(vec![5.0, 63.0, 53.0], identity)
        );
    }

    // let data = get_training_data("train/complex_polynomial_2.txt");
    // let (data, min_x, max_x) = normalize(data);
    //
    // let (x, y) = (
    //     data[0..data.len() - 1].to_vec(),
    //     data.last().unwrap().to_vec(),
    // );
    // let cols = x[0].len();
    // let rows = x.len();
    // let final_x: Vec<f64> = x.into_iter().flatten().collect();
    // let x = Matrix::with_vector(final_x, cols, rows);
    // let y = Matrix::with_vector(y, cols, 1);
    // println!("{}", x);
    // let mut model = PolynomialRegression::new(3, 3);
    // model.fit(&x, &y, 0.00000005, 1000 * 1000 * 10, identity);
    // model.dump();
    // println!(
    //     "{} should be (95)",
    //     // model.predict(vec![22.0, 22.0], identity)
    //     model.predict(
    //         vec![
    //             (1.0 - min_x) / (max_x - min_x),
    //             (2.0 - min_x) / (max_x - min_x),
    //             (3.0 - min_x) / (max_x - min_x)
    //         ],
    //         identity
    //     )
    // );
}
