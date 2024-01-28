use std::fs;

fn get_training_data(mut path: &str) -> Vec<Vec<f64>> {
    let contents = fs::read_to_string(&mut path).unwrap();
    let mut training_data = vec![];
    for line in contents.lines() {
        let line: Vec<f64> = line.split(' ').map(|x| x.parse::<f64>().unwrap()).collect();
        training_data.push(line);
    }
    return training_data;
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-value))
}

fn identity(value: f64) -> f64 {
    value
}

fn cost<F>(training_data: &Vec<Vec<f64>>, w: &[f64], b: f64, activation_function: F) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut mse = 0.0;
    for data in training_data.iter() {
        let (xs, y) = (&data[0..data.len() - 1], data.last().unwrap());
        let mut predicted_result = b;
        for (i, x) in xs.iter().enumerate() {
            predicted_result += x * w[i];
        }
        predicted_result = activation_function(predicted_result);
        let actual_result = y;
        mse += f64::powi(actual_result - predicted_result, 2);
    }
    mse /= training_data.len() as f64;
    mse
}

fn train<F>(
    path: &str,
    learning_rate: f64,
    mut weights: Vec<f64>,
    mut bias: f64,
    epochs: usize,
    activation_function: F,
) -> (Vec<f64>, f64)
where
    F: Fn(f64) -> f64,
{
    let training_data = get_training_data(path);
    let eps = 0.001;
    for _ in 0..epochs {
        let c = cost(&training_data, &weights, bias, &activation_function);
        let db = (cost(&training_data, &weights, bias + eps, &activation_function) - c) / eps;
        for (i, _) in weights.clone().iter().enumerate() {
            let mut tmp_weights = weights.clone();
            tmp_weights[i] += eps;
            let bc = cost(&training_data, &tmp_weights, bias, &activation_function);
            let dw = (bc - c) / eps;
            weights[i] -= learning_rate * dw;
        }
        bias -= learning_rate * db;
        // println!(
        //     "MSE = {:?}",
        //     cost(&training_data, &weights, bias, &activation_function)
        // );
    }
    println!(
        "MSE = {:?}",
        cost(&training_data, &weights, bias, activation_function)
    );
    (weights, bias)
}

fn main() {
    let (weights, bias) = train(
        "train/double_plus_one.txt",
        0.001,
        vec![10.0],
        7.0,
        100_000_0,
        identity,
    );
    let y = 500.0 * weights[0] + bias;
    println!("w = {}, b = {bias} y = {y}\n", weights[0]);

    let (weights, bias) = train(
        "train/and_gate.txt",
        0.1,
        vec![0.42, 0.87],
        0.52,
        100_000_0,
        sigmoid,
    );
    for i in 0..2 {
        for j in 0..2 {
            let y = sigmoid(i as f64 * weights[0] + j as f64 * weights[1] + bias);
            println!("{i} & {j} = {}", y)
        }
    }
    println!();
    let (weights, bias) = train(
        "train/or_gate.txt",
        0.1,
        vec![0.42, 0.87],
        0.52,
        100_000_0,
        sigmoid,
    );
    for i in 0..2 {
        for j in 0..2 {
            let y = sigmoid(i as f64 * weights[0] + j as f64 * weights[1] + bias);
            println!("{i} & {j} = {}", y)
        }
    }
    println!();
}
