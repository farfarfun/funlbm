import numpy as np


def to_sparse_matrix(dense_matrix):
    non_zero_indices = np.nonzero(dense_matrix)
    non_zero_values = dense_matrix[non_zero_indices]
    sparse_matrix = np.vstack((non_zero_indices[0], non_zero_indices[1], non_zero_indices[2], non_zero_values)).T
    return sparse_matrix


def write_to_tecplot(dense_matrix, filename):
    sparse_matrix = to_sparse_matrix(dense_matrix)
    shape = dense_matrix.shape
    with open(filename, "w") as f:
        f.write('TITLE = "Sparse Matrix Data"\n')
        f.write('VARIABLES = "Z", "Y", "X", "Value"\n')
        f.write(f"ZONE I={shape[2]}, J={shape[1]}, K={shape[0]}, F=POINT\n")

        for row in sparse_matrix:
            f.write(f"{row[0]} {row[1]} {row[2]} {row[3]}\n")


def example():
    # 示例用法
    dense_matrix = np.array([[[0, 1, 0], [2, 0, 3]], [[0, 4, 0], [5, 0, 6]]])
    write_to_tecplot(dense_matrix, "sparse_matrix.dat")


# example()
