use tnn::{
    linear_algebra::par_mult::{mult_par_strassen, Matrix},
    linear_regression::*,
    model::Model,
    utils::{get_training_data, sigmoid},
};

fn main() {
    let mat1 = Matrix::with_vector(vec![1.0, 2.0, 3.0], 3, 1);
    let mat2 = Matrix::with_vector(vec![2.0, 2.0, 2.0], 3, 1);

    let mat3 = mult_par_strassen(&mat1, &mat2);
    println!("{:?}", mat3);

    // let mut linear_reg = LinearRegression::new(1);
    // let mut data = get_training_data("train/will_pass.txt");
    // linear_reg.fit(&mut data, 0.1, 1000 * 1000 * 10, sigmoid);
    // linear_reg.dump();
    // println!("{}", linear_reg.predict(vec![5.08], sigmoid));

    // let data = get_training_data("train/complex_polynomial.txt");
    // let (mut normalized_data, min_x, max_x) = normalize(data);
    //
    // let mut model = PolynomialRegression::new(3);
    // model.fit(&mut normalized_data, 0.004, 100_000_000, identity);
    // let x = (50.0 - min_x) / (max_x - min_x);
    // println!("prediction: {}", model.predict(vec![x], identity));
    // model.dump();
}
