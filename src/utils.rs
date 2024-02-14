use std::fs;

pub fn get_training_data(mut path: &str) -> Vec<Vec<f64>> {
    let contents = fs::read_to_string(&mut path).unwrap();
    let line = contents.lines().next().unwrap();
    let rows = contents.lines().count();
    let cols = line.chars().filter(|ch| *ch == ' ').count() + 1;

    let mut training_data = vec![vec![0.0; cols]; rows];
    for (j, line) in contents.lines().enumerate() {
        for i in 0..cols {
            let line: Vec<f64> = line.split(' ').map(|x| x.parse::<f64>().unwrap()).collect();
            training_data[j][i] = line[i];
        }
    }
    return training_data;
}

pub fn minf(f: f64, s: f64) -> f64 {
    if f <= s {
        return f;
    } else {
        s
    }
}
pub fn maxf(f: f64, s: f64) -> f64 {
    if f >= s {
        return f;
    } else {
        s
    }
}

pub fn standardize(data: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let num_features = data[0].len();
    let mut means = vec![0.0; num_features];
    let mut std_devs = vec![0.0; num_features];

    // Calculate means
    for row in &data {
        for (i, &value) in row.iter().rev().skip(1).enumerate() {
            means[i] += value;
        }
    }

    for mean in &mut means {
        *mean /= data.len() as f64;
    }

    // Calculate standard deviations
    for row in &data {
        for (i, &value) in row.iter().rev().skip(1).enumerate() {
            std_devs[i] += (value - means[i]).powi(2);
        }
    }

    for std_dev in &mut std_devs {
        *std_dev = (*std_dev / (data.len() - 1) as f64).sqrt();
    }

    // Standardize the data
    let mut standardized_data = vec![];

    for row in &data {
        let standardized_row: Vec<f64> = row
            .iter()
            .enumerate()
            .map(|(i, &value)| (value - means[i]) / std_devs[i])
            .collect();

        standardized_data.push(standardized_row);
    }

    standardized_data
}

pub fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-value))
}

pub fn identity(value: f64) -> f64 {
    value
}
