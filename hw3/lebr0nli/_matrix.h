#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

class Matrix {
public:
    Matrix(size_t rows, size_t cols);
    Matrix(const Matrix &other);                // Copy constructor
    Matrix(Matrix &&other) noexcept;            // Move constructor
    Matrix &operator=(const Matrix &other);     // Copy assignment
    Matrix &operator=(Matrix &&other) noexcept; // Move assignment
    ~Matrix() = default;

    double &operator()(size_t row, size_t col);
    const double &operator()(size_t row, size_t col) const;
    bool operator==(const Matrix &other) const;

    size_t nrow() const { return rows_; }
    size_t ncol() const { return cols_; }

private:
    std::unique_ptr<double[]> data_;
    size_t rows_;
    size_t cols_;
};

double Matrix_getitem(const Matrix &m, const std::pair<size_t, size_t> &indices);
void Matrix_setitem(Matrix &m, const std::pair<size_t, size_t> &indices, double value);

Matrix multiply_naive(const Matrix &a, const Matrix &b);
Matrix multiply_tile(const Matrix &a, const Matrix &b, size_t tile_size);
Matrix multiply_mkl(const Matrix &a, const Matrix &b);

#endif // _MATRIX_H_
