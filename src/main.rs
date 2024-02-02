use tnn::{
    linear_algebra::par_mult::{mult_naive, mult_par_strassen, mult_transpose, Matrix},
    linear_regression::*,
    model::Model,
    utils::{get_training_data, identity, normalize, sigmoid},
};

fn main() {
    let mut linear_reg = LinearRegression::new(3);
    let data = get_training_data("train/complex_add.txt");
    let (data, min_x, max_x) = normalize(data);
    let (x, y) = (
        data[0..data.len() - 1].to_vec(),
        data.last().unwrap().to_vec(),
    );
    let cols = x[0].len();
    let rows = x.len();
    let mut final_x = vec![];
    for i in 0..cols {
        let mut tmp = vec![];
        for j in 0..rows {
            tmp.push(x[j][i])
        }
        final_x.push(tmp);
    }
    let final_x: Vec<f64> = final_x.into_iter().flatten().collect();
    let x = Matrix::with_vector(final_x, cols, rows);
    let y = Matrix::with_vector(y, cols, 1);
    println!("{}", x);
    println!("{}", y);
    linear_reg.fit(&x, &y, 0.0001, 1000 * 1000 * 10, identity);
    linear_reg.dump();
    println!(
        "22 + 2 * 22 + 3 * 22  = {} (should be 132)",
        linear_reg.predict(
            vec![
                (22.0 - min_x) / (max_x - min_x),
                (22.0 - min_x) / (max_x - min_x),
                (22.0 - min_x) / (max_x - min_x)
            ],
            identity
        )
    );
    // println!("0 & 1 = {}", linear_reg.predict(vec![0.0, 1.0], sigmoid));
    // println!("1 & 0 = {}", linear_reg.predict(vec![1.0, 0.0], sigmoid));
    // println!("1 & 1 = {}", linear_reg.predict(vec![1.0, 1.0], sigmoid));

    // let mut linear_reg = LinearRegression::new(1);
    // let data = get_training_data("train/will_pass.txt");
    // println!("{data:?}");
    // let normalized_data = normalize(data);
    // println!("---------------------");
    // println!("{:?}", normalized_data.0);
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
