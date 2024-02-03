use std::fmt;

/* Floating-point comparison precision */
const EPSILON: f64 = 0.000001;

/**
 * Basic matrix implementation. Metadata with a vector of f64 floats; matrix
 * elements are indexed into the single vector.
 */
#[derive(Debug, Clone)]
pub struct Matrix {
    // Number of rows
    pub rows: usize,

    // Number of columns
    pub cols: usize,

    // Number of elements; rows * cols
    pub n: usize,

    // Underlying matrix data
    pub elements: Vec<f64>,
}

impl Matrix {
    /**
     * Initializes a new Matrix, wrapping the provided vector. Rows and cols must be nonzero.
     */
    pub fn with_vector(elements: Vec<f64>, rows: usize, cols: usize) -> Matrix {
        if rows == 0 || cols == 0 {
            panic!("cannot create a null matrix");
        }

        Matrix {
            rows,
            cols,
            n: rows * cols,
            elements,
        }
    }

    /**
     * Creates a new empty matrix of the specified dimensions; underlying vector is initialized to zeroes.
     * Rows and cols must be nonzero.
     */
    pub fn new(rows: usize, cols: usize) -> Matrix {
        if rows == 0 || cols == 0 {
            panic!("cannot create a null matrix");
        }

        return Matrix::with_vector(vec![0.0; rows * cols], rows, cols);
    }

    /**
     * Returns a deep copy of this Matrix; clones underlying vector data as well.
     */
    pub fn copy(&self) -> Matrix {
        return Matrix::with_vector(self.elements.to_vec(), self.rows, self.cols);
    }

    /**
     * Returns the element at (i, j).
     * Unsafe.
     */
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        unsafe {
            return *self.elements.get_unchecked(i * self.cols + j);
        }
    }

    #[inline]
    pub fn get_element(&self, row: usize, col: usize) -> f64 {
        self.elements[row * self.cols + col]
    }

    #[inline]
    pub fn set_element(&mut self, row: usize, col: usize, value: f64) {
        self.elements[row * self.cols + col] = value;
    }

    /**
     * Returns true if the number of rows in this Matrix equals the number of columns.
     */
    #[inline]
    pub fn is_square(&self) -> bool {
        return self.rows == self.cols;
    }

    /**
     * Adds the contents of `b` to this Matrix, and returns self. Panics if `b` is not the same size as this Matrix.
     */
    pub fn add(&mut self, b: &Matrix) -> &mut Matrix {
        if self.rows == b.rows && self.cols == b.cols {
            for i in 0..self.n {
                self.elements[i] += b.elements[i];
            }
        } else {
            panic!(
                "matrices are not the same size! A: [{}, {}], B: [{}, {}]",
                self.rows, self.cols, b.rows, b.cols
            );
        }

        return self;
    }

    /**
     * Subtracts the contents of `b` from this Matrix, and returns self. Panics if `b` is not the same size as this Matrix.
     */
    pub fn sub(&mut self, b: &Matrix) -> &mut Matrix {
        if self.rows == b.rows && self.cols == b.cols {
            for i in 0..self.n {
                self.elements[i] -= b.elements[i];
            }
        } else {
            panic!(
                "matrices are not the same size! A: [{}, {}], B: [{}, {}]",
                self.rows, self.cols, b.rows, b.cols
            );
        }

        return self;
    }

    /**
     * Returns true if all elements of `b` are equal to all elements of this Matrix, subject to a difference of value of `EPSILON` or greater.
     * Panics if `b` is not the same size as this Matrix.
     */
    pub fn eq(&self, b: &Matrix) -> bool {
        if self.rows == b.rows && self.cols == b.cols {
            for i in 0..self.n {
                let delta = (self.elements[i] - b.elements[i]).abs();
                if delta > EPSILON {
                    return false;
                }
            }

            return true;
        } else {
            panic!(
                "matrices are not the same size! A: [{}, {}], B: [{}, {}]",
                self.rows, self.cols, b.rows, b.cols
            );
        }
    }

    /**
     * Returns a new Matrix with the contents of this Matrix copied into it. If the provided parameter `n` is
     * smaller or equal to the existing number of rows and columns of this matrix, a copy of this matrix
     * is returned.
     *
     * Otherwise, new a square n x n Matrix is returned, with any added rows or columns filled with zeroes.
     */
    pub fn pad(&self, n: usize) -> Matrix {
        if n <= self.rows && n <= self.cols {
            return self.copy();
        } else {
            // Initialize new vector with expected capacity
            let mut v: Vec<f64> = Vec::with_capacity(n * n);

            for i in 0..self.rows {
                for j in 0..self.cols {
                    v.push(self.at(i, j));
                }

                // These are additional, padded columns
                for _ in self.cols..n {
                    v.push(0.0);
                }
            }

            // These are additional, padded rows
            for _ in self.rows..n {
                for _ in 0..n {
                    v.push(0.0);
                }
            }

            return Matrix::with_vector(v, n, n);
        }
    }

    /**
     * Returns a new Matrix which contains a copy of this matrix, reduced to the given
     * number of rows and columns, starting from index [0, 0]. Typically used to remove
     * padding applied during the beginning of Strassen multiplication, to return a matrix
     * back to its original dimensions.
     * Panics if the specified number of rows or columns are larger than this Matrix's number of rows or columns.
     */
    pub fn reduce(&self, rows: usize, cols: usize) -> Matrix {
        if rows > self.rows || cols > self.cols {
            panic!("Tried to reduce self to larger dimensions");
        }

        let mut v: Vec<f64> = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            let indx = i * self.cols;
            v.extend_from_slice(&self.elements[indx..(indx + cols)])
        }

        return Matrix::with_vector(v, rows, cols);
    }

    /**
     * Returns a new Matrix containing the transpose of this Matrix.
     */
    pub fn transpose(&self) -> Matrix {
        let mut v: Vec<f64> = Vec::with_capacity(self.n);

        for i in 0..self.cols {
            for j in 0..self.rows {
                v.push(self.at(j, i));
            }
        }

        return Matrix::with_vector(v, self.cols, self.rows);
    }

    pub fn dot(&self, b: &Matrix) -> f64 {
        assert!(self.rows == b.rows);
        assert!(self.cols == b.cols);

        let mut res = 0.0;

        let m = self.rows;
        let c = self.cols;

        for i in 0..m {
            for j in 0..c {
                res += self.at(i, j) * b.at(i, j);
            }
        }

        res
    }

    pub fn prod_element_wise(&mut self, value: f64) {
        let m = self.rows;
        let c = self.cols;

        for i in 0..m {
            for j in 0..c {
                let cur = self.at(i, j);
                self.set_element(i, j, cur * value);
            }
        }
    }

    pub fn add_element_wise(&mut self, value: f64) {
        let m = self.rows;
        let c = self.cols;

        for i in 0..m {
            for j in 0..c {
                let cur = self.at(i, j);
                self.set_element(i, j, cur + value);
            }
        }
    }

    pub fn divide_element_wise(&mut self, value: f64) {
        assert!(value != 0.0);

        let m = self.rows;
        let c = self.cols;

        for i in 0..m {
            for j in 0..c {
                let cur = self.at(i, j);
                self.set_element(i, j, cur / value);
            }
        }
    }

    /**
     * Multiplies this Matrix by `b`, using the provided `multiplier` function.
     */
    pub fn mult(&self, b: &Matrix, multipler: fn(&Matrix, &Matrix) -> Matrix) -> Matrix {
        return multipler(self, b);
    }
}

/**
 * Pretty-printing for the Matrix struct.
 */
impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Matrix {} X {}\n[\n", self.rows, self.cols).unwrap();
        for i in 0..self.rows {
            write!(f, "\t").unwrap();
            for j in 0..self.cols {
                write!(f, "{:.5}    ", self.elements[i + j * self.rows]).unwrap();
                // write!(f, "{:.5}    ", self.at(i, j)).unwrap();
            }
            write!(f, "\n").unwrap();
        }
        write!(f, "]\n",)
    }
}
