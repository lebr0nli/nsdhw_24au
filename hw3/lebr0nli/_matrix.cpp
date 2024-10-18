#include <algorithm>
#include <cstring>
#include <mkl.h>
#include <stdexcept>
#include <utility>

#include "_matrix.h"

Matrix::Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
    data_ = std::make_unique<double[]>(rows * cols);
}

Matrix::Matrix(const Matrix &other) : rows_(other.rows_), cols_(other.cols_) {
    data_ = std::make_unique<double[]>(rows_ * cols_);
    std::memcpy(data_.get(), other.data_.get(), rows_ * cols_ * sizeof(double));
}

Matrix::Matrix(Matrix &&other) noexcept
    : data_(std::move(other.data_)), rows_(other.rows_), cols_(other.cols_) {
    other.rows_ = other.cols_ = 0;
}

Matrix &Matrix::operator=(const Matrix &other) {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = std::make_unique<double[]>(rows_ * cols_);
        std::memcpy(data_.get(), other.data_.get(),
                    rows_ * cols_ * sizeof(double));
    }
    return *this;
}

Matrix &Matrix::operator=(Matrix &&other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        rows_ = other.rows_;
        cols_ = other.cols_;
        other.rows_ = other.cols_ = 0;
    }
    return *this;
}

double &Matrix::operator()(size_t row, size_t col) {
    return data_[row * cols_ + col];
}

const double &Matrix::operator()(size_t row, size_t col) const {
    return data_[row * cols_ + col];
}

bool Matrix::operator==(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_)
        return false;
    return std::memcmp(data_.get(), other.data_.get(),
                       rows_ * cols_ * sizeof(double)) == 0;
}

double Matrix_getitem(const Matrix &m, const std::pair<size_t, size_t> &indices) {
    size_t row = indices.first;
    size_t col = indices.second;
    if (row >= m.nrow() || col >= m.ncol()) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return m(row, col);
}

void Matrix_setitem(Matrix &m, const std::pair<size_t, size_t> &indices, double value) {
    size_t row = indices.first;
    size_t col = indices.second;
    if (row >= m.nrow() || col >= m.ncol()) {
        throw std::out_of_range("Matrix indices out of range");
    }
    m(row, col) = value;
}

Matrix multiply_naive(const Matrix &a, const Matrix &b) {
    if (a.ncol() != b.nrow()) {
        throw std::invalid_argument(
            "Matrix dimensions do not match for multiplication");
    }

    Matrix result(a.nrow(), b.ncol());
    for (size_t i = 0; i < a.nrow(); ++i) {
        for (size_t j = 0; j < b.ncol(); ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < a.ncol(); ++k) {
                sum += a(i, k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

Matrix multiply_tile(const Matrix &mat_A, const Matrix &mat_B, size_t tile_size) {
    if (mat_A.ncol() != mat_B.nrow()) {
        throw std::invalid_argument(
            "Matrix dimensions do not match for multiplication");
    }

    Matrix result(mat_A.nrow(), mat_B.ncol());

    // Initialize result matrix with zeros
    for (size_t i = 0; i < result.nrow(); ++i) {
        for (size_t j = 0; j < result.ncol(); ++j) {
            result(i, j) = 0.0;
        }
    }

    for (size_t i = 0; i < mat_A.nrow(); i += tile_size) {
        for (size_t k = 0; k < mat_A.ncol(); k += tile_size) {
            for (size_t j = 0; j < mat_B.ncol(); j += tile_size) {
                size_t i_end = std::min(mat_A.nrow(), i + tile_size);
                size_t k_end = std::min(mat_A.ncol(), k + tile_size);
                size_t j_end = std::min(mat_B.ncol(), j + tile_size);

                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t kk = k; kk < k_end; ++kk) {
                        double r = mat_A(ii, kk);
                        for (size_t jj = j; jj < j_end; ++jj) {
                            result(ii, jj) += r * mat_B(kk, jj);
                        }
                    }
                }
            }
        }
    }
    return result;
}

Matrix multiply_mkl(const Matrix &a, const Matrix &b) {
    if (a.ncol() != b.nrow()) {
        throw std::invalid_argument(
            "Matrix dimensions do not match for multiplication");
    }

    Matrix result(a.nrow(), b.ncol());
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a.nrow(), b.ncol(),
                a.ncol(), 1.0, &a(0, 0), a.ncol(), &b(0, 0), b.ncol(), 0.0,
                &result(0, 0), result.ncol());
    return result;
}
