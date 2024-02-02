pub use crate::linear_algebra::matrix::Matrix;
use rand::thread_rng;

static mut TEST_STATE: bool = false;

/**
 * Variant of the naive multiplication algorithm, which uses the transpose of `b`, resuting in better memory locality performance characteristics. Still O(n^3).
 * Panics if matrices `a` and `b` are of incompatbile dimensions.
 */
pub fn mult_transpose(a: &Matrix, b: &Matrix) -> Matrix {
    if a.cols == b.rows {
        let m = a.rows;
        let n = b.cols;
        let p = a.cols;
        let t = b.transpose();
        let mut c: Vec<f64> = Vec::with_capacity(m * n);

        for i in 0..m {
            for j in 0..n {
                let mut sum: f64 = 0.0;
                for k in 0..p {
                    sum += a.at(i, k) * t.at(j, k);
                }

                c.push(sum);
            }
        }

        return Matrix::with_vector(c, m, n);
    } else {
        panic!("Matrix sizes do not match");
    }
}

/**
 * Strassen algorithm. See https://en.wikipedia.org/wiki/Strassen_algorithm
 * Breaks the provided matrices down into 7 smaller submatrices for multiplication, which results in
 * smaller asymptotic complexity of around O(n^2.8), at the expense of a higher scalar constant due to the extra work required.
 * Falls back to the transpose naive multiplication method if row and column dimensions are 64 or less.
 * Recurses as input matrices are broken down and this algorithm is run further on those submatrices.
 * Panics if matrices `a` and `b` are of incompatbile dimensions.
 */
pub fn mult_strassen(a: &Matrix, b: &Matrix) -> Matrix {
    if a.cols != b.rows {
        panic!("Matrix sizes do not match");
    }

    let mut max_dimension = a.rows;

    // Find the largest row or column dimension across `a` and `b`
    if a.cols >= a.rows && a.cols >= b.cols {
        max_dimension = a.cols;
    } else if b.rows >= b.cols && b.rows >= a.rows {
        max_dimension = b.rows;
    } else if b.cols >= b.rows && b.cols >= a.cols {
        max_dimension = b.cols;
    }

    // If the largest dimension is odd, we'll add one and then pad the matrix to make it
    // an even number of rows and columns
    if max_dimension % 2 == 1 {
        max_dimension += 1;
    }

    if a.is_square() && b.is_square() && a.rows == max_dimension {
        // The matrices are square; proceed
        return _mult_strassen(&a, &b);
    } else {
        // Pad `a` and `b` to `max_dimension` and pass to underlying function `_mult_strassen`. Strip out
        // extra padded rows and columns after that operation is complete.
        return _mult_strassen(&a.pad(max_dimension), &b.pad(max_dimension)).reduce(a.rows, a.rows);
    }
}

/**
 * Inner Strassen algorithm logic.
 */
fn _mult_strassen(a: &Matrix, b: &Matrix) -> Matrix {
    unsafe {
        // Ugly hack for enabling recursion testing in unit tests.
        // If not test state, fall back to transpose matrix multiplication if
        // input Matrix rows and columns are 64 or less.
        if (!TEST_STATE && a.rows <= 64) || (TEST_STATE && a.rows <= 2) {
            return mult_transpose(a, b);
        }
    }

    /* This will be the row and column size of the submatrices */
    let m = a.rows / 2;

    /* Top left submatrix */
    let tl_row_start = 0;
    let tl_col_start = 0;

    /* Top right submatrix */
    let tr_row_start = 0;
    let tr_col_start = m;

    /* Bottom left submatrix */
    let bl_row_start = m;
    let bl_col_start = 0;

    /* Bottom right submatrix */
    let br_row_start = m;
    let br_col_start = m;

    /* Vectors for 7 submatrices of `a` */
    let mut aa1 = Vec::with_capacity(m * m);
    let mut aa2 = Vec::with_capacity(m * m);
    let mut aa3 = Vec::with_capacity(m * m);
    let mut aa4 = Vec::with_capacity(m * m);
    let mut aa5 = Vec::with_capacity(m * m);
    let mut aa6 = Vec::with_capacity(m * m);
    let mut aa7 = Vec::with_capacity(m * m);

    /* Vectors for 7 submatrices of `b` */
    let mut bb1 = Vec::with_capacity(m * m);
    let mut bb2 = Vec::with_capacity(m * m);
    let mut bb3 = Vec::with_capacity(m * m);
    let mut bb4 = Vec::with_capacity(m * m);
    let mut bb5 = Vec::with_capacity(m * m);
    let mut bb6 = Vec::with_capacity(m * m);
    let mut bb7 = Vec::with_capacity(m * m);

    /*
     * The output matrix C is expressed in terms of the block matrices M1..M7
     *
     * C1,1 = M1 + M4 - M5 + M7
     * C1,2 = M3 + M5
     * C2,1 = M2 + M4
     * C2,2 = M1 - M2 + M3 + M6
     *
     * Each of the block matrices M1..M7 is composed of products of quadrants from A and B as follows:
     *
     * M1 = AA[0] * BB[0] = (A1,1 + A2,2)(B1,1 + B2,2)
     * M2 = AA[1] * BB[1] = (A2,1 + A2,2)(B1,1)
     * M3 = AA[2] * BB[2] = (A1,1)(B1,2 - B2,2)
     * M4 = AA[3] * BB[3] = (A2,2)(B2,1 - B1,1)
     * M5 = AA[4] * BB[4] = (A1,1 + A1,2)(B2,2)
     * M6 = AA[5] * BB[5] = (A2,1 - A1,1)(B1,1 + B1,2)
     * M7 = AA[6] * BB[6] = (A1,2 - A2,2)(B2,1 + B2,2)
     */

    /* Initializes submatrices of `a` based on its quadrants, the manner described below */

    /* AA1 = (A1,1 + A2,2) */
    submatrix_add(
        &mut aa1,
        a,
        tl_row_start,
        tl_col_start,
        br_row_start,
        br_col_start,
        m,
    );
    /* AA2 = (A2,1 + A2,2) */
    submatrix_add(
        &mut aa2,
        a,
        bl_row_start,
        bl_col_start,
        br_row_start,
        br_col_start,
        m,
    );
    /* AA3 = (A1,1) */
    submatrix_cpy(&mut aa3, a, tl_row_start, tl_col_start, m);
    /* AA4 = (A2,2) */
    submatrix_cpy(&mut aa4, a, br_row_start, br_col_start, m);
    /* AA5 = (A1,1 + A1,2) */
    submatrix_add(
        &mut aa5,
        a,
        tl_row_start,
        tl_col_start,
        tr_row_start,
        tr_col_start,
        m,
    );
    /* AA6 = (A2,1 - A1,1) */
    submatrix_sub(
        &mut aa6,
        a,
        bl_row_start,
        bl_col_start,
        tl_row_start,
        tl_col_start,
        m,
    );
    /* AA7 = (A1,2 - A2,2) */
    submatrix_sub(
        &mut aa7,
        a,
        tr_row_start,
        tr_col_start,
        br_row_start,
        br_col_start,
        m,
    );

    /* Initializes submatrices of `b` based on its quadrants, the manner described below */

    /* BB1 = (B1,1 + B2,2) */
    submatrix_add(
        &mut bb1,
        b,
        tl_row_start,
        tl_col_start,
        br_row_start,
        br_col_start,
        m,
    );
    /* BB2 = (B1,1) */
    submatrix_cpy(&mut bb2, b, tl_row_start, tl_col_start, m);
    /* BB3 = (B1,2 - B2,2) */
    submatrix_sub(
        &mut bb3,
        b,
        tr_row_start,
        tr_col_start,
        br_row_start,
        br_col_start,
        m,
    );
    /* BB4 = (B2,1 - B1,1) */
    submatrix_sub(
        &mut bb4,
        b,
        bl_row_start,
        bl_col_start,
        tl_row_start,
        tl_col_start,
        m,
    );
    /* BB5 = (B2,2) */
    submatrix_cpy(&mut bb5, b, br_row_start, br_col_start, m);
    /* BB6 = (B1,1 + B1,2) */
    submatrix_add(
        &mut bb6,
        b,
        tl_row_start,
        tl_col_start,
        tr_row_start,
        tr_col_start,
        m,
    );
    /* BB7 = (B2,1 + B2,2) */
    submatrix_add(
        &mut bb7,
        b,
        bl_row_start,
        bl_col_start,
        br_row_start,
        br_col_start,
        m,
    );

    /*
     * Build the intermediate matrices M1..M7
     *
     * The following operations each recurse further, passing their respective submatrices into the
     * main `mult_strassen` function above.
     */

    let mut m1 = mult_strassen(
        &mut Matrix::with_vector(aa1, m, m),
        &mut Matrix::with_vector(bb1, m, m),
    );

    let m2 = mult_strassen(
        &mut Matrix::with_vector(aa2, m, m),
        &mut Matrix::with_vector(bb2, m, m),
    );

    let m3 = mult_strassen(
        &mut Matrix::with_vector(aa3, m, m),
        &mut Matrix::with_vector(bb3, m, m),
    );

    let mut m4 = mult_strassen(
        &mut Matrix::with_vector(aa4, m, m),
        &mut Matrix::with_vector(bb4, m, m),
    );

    let mut m5 = mult_strassen(
        &mut Matrix::with_vector(aa5, m, m),
        &mut Matrix::with_vector(bb5, m, m),
    );

    let m6 = mult_strassen(
        &mut Matrix::with_vector(aa6, m, m),
        &mut Matrix::with_vector(bb6, m, m),
    );

    let mut m7 = mult_strassen(
        &mut Matrix::with_vector(aa7, m, m),
        &mut Matrix::with_vector(bb7, m, m),
    );

    /* C1,1 = M1 + M4 - M5 + M7 */
    let m11 = m7.sub(&m5).add(&m4).add(&m1);

    /* C1,2 = M3 + M5 */
    let m12 = m5.add(&m3);

    /* C2,1 = M2 + M4 */
    let m21 = m4.add(&m2);

    /* C2,2 = M1 - M2 + M3 + M6 */
    let m22 = m1.sub(&m2).add(&m3).add(&m6);

    /* Return a single matrix composing each of these four matrices as quadrants */
    return reconstitute(&m11, &m12, &m21, &m22, m, a.rows);
}

/**
 * Adds the two specified submatrices of `a` into `c`, using the provided quadrant offsets.
 * `a_row_start` and `a_col_start` refer to row and column offsets into `a`, used to represent a matrix quadrant.
 * Similarly for the `b` values.
 * Quadrants have `m` rows and columns.
 */
pub fn submatrix_add(
    c: &mut Vec<f64>,
    a: &Matrix,
    a_row_start: usize,
    a_col_start: usize,
    b_row_start: usize,
    b_col_start: usize,
    m: usize,
) {
    for i in 0..m {
        for j in 0..m {
            c.push(a.at(a_row_start + i, a_col_start + j) + a.at(b_row_start + i, b_col_start + j))
        }
    }
}

/**
 * Subtracts the two specified submatrices of `a` into `c`, using the provided quadrant offsets.
 * `a_row_start` and `a_col_start` refer to row and column offsets into `a`, used to represent a matrix quadrant.
 * Similarly for the `b` values.
 * Quadrants have `m` rows and columns.
 */
pub fn submatrix_sub(
    c: &mut Vec<f64>,
    a: &Matrix,
    a_row_start: usize,
    a_col_start: usize,
    b_row_start: usize,
    b_col_start: usize,
    m: usize,
) {
    for i in 0..m {
        for j in 0..m {
            c.push(a.at(a_row_start + i, a_col_start + j) - a.at(b_row_start + i, b_col_start + j))
        }
    }
}

/**
 * Copies the specified submatrix of `a` into `c`, using the provided quadrant offsets.
 * `a_row_start` and `a_col_start` refer to row and column offsets into `a`, used to represent a matrix quadrant.
 * Quadrants have `m` rows and columns.
 */
pub fn submatrix_cpy(
    c: &mut Vec<f64>,
    a: &Matrix,
    a_row_start: usize,
    a_col_start: usize,
    m: usize,
) {
    for i in 0..m {
        let indx = ((a_row_start + i) * a.cols) + a_col_start;
        c.extend_from_slice(&a.elements[indx..(indx + m)]);
    }
}

/**
 * Reconstitutes a large matrix composed of the four provided matrices, composing
 * them as quadrants in a larger matrix.
 * `m11` refers to `M(1,1)` for example.
 */
pub fn reconstitute(
    m11: &Matrix,
    m12: &Matrix,
    m21: &Matrix,
    m22: &Matrix,
    m: usize,
    n: usize,
) -> Matrix {
    let mut v: Vec<f64> = Vec::with_capacity(n * n);
    let mut indx: usize;

    for i in 0..m {
        indx = i * m;
        v.extend_from_slice(&m11.elements[indx..(indx + m)]);
        v.extend_from_slice(&m12.elements[indx..(indx + m)]);
    }

    for i in 0..m {
        indx = i * m;
        v.extend_from_slice(&m21.elements[indx..(indx + m)]);
        v.extend_from_slice(&m22.elements[indx..(indx + m)]);
    }

    return Matrix::with_vector(v, n, n);
}
