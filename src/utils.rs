use std::fs;

pub fn get_training_data(mut path: &str) -> Vec<Vec<f64>> {
    let contents = fs::read_to_string(&mut path).unwrap();
    let mut training_data = vec![];
    for line in contents.lines() {
        let line: Vec<f64> = line.split(' ').map(|x| x.parse::<f64>().unwrap()).collect();
        training_data.push(line);
    }
    return training_data;
}

fn minf(f: f64, s: f64) -> f64 {
    if f <= s {
        return f;
    } else {
        s
    }
}
fn maxf(f: f64, s: f64) -> f64 {
    if f >= s {
        return f;
    } else {
        s
    }
}

pub fn normalize(data: Vec<Vec<f64>>) -> (Vec<Vec<f64>>, f64, f64) {
    println!("Normalizing {data:#?}");
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    for row in data.iter() {
        max_x = maxf(max_x, row[0]);
        max_y = maxf(max_y, row[1]);
        min_x = minf(min_x, row[0]);
        min_y = minf(min_y, row[1]);
    }
    println!("{max_x}, {max_y}, {min_x}, {min_y}");
    let mut normalized_data = vec![];
    for row in data.iter() {
        let (x, y) = (row[0], row[1]);
        normalized_data.push(vec![(x - min_x) / (max_x - min_x), row[1]])
    }
    (normalized_data, min_x, max_x)
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
