# @author: František Dostál
import numpy as np

# COLUMN = minimizing
# ROW = maximizing

def evaluate(matrix_r, matrix_c, row_strategy, column_strategy):
    return row_strategy @ matrix_r @ column_strategy, row_strategy @ matrix_c @ column_strategy

def evaluate_zero_sum(matrix, row_strategy, column_strategy):
    return row_strategy @ matrix @ column_strategy

#algorithms for zero sum games
#opponents best response, not equilibrial response
def best_response_calc_row(matrix, row_strategy):
    return np.argmin((row_strategy @ matrix))

#opponents best response to given strategy, not equilibrial response
def best_response_calc_column(matrix, column_strategy):
    return np.argmin(-( matrix @ column_strategy))


def best_response_value_column(matrix, column_strategy):
    #column value against best responding opponent
    return np.min(-matrix @ column_strategy)


def best_response_value_row(matrix, row_strategy):
    #row value against best responding opponent
    return np.min(row_strategy @ matrix)

def strongly_dominated_row(matrix):
    for index in range(matrix.shape[0]):
        c = np.delete(matrix, index, 0)
        if np.any(np.all(c > matrix[index], axis=1)):
            return index
    return None

def strongly_dominated_column(matrix):
    for index in range(matrix.shape[1]):
        c = np.delete(matrix, index, 1)
        if np.any(np.all((c.transpose() < matrix[:, index]).transpose(), axis=0)):
            return index
    return None

def strongly_dominated_rows(matrix):
    #generator version
    M = matrix.copy()
    for index in range(M.shape[0]):
        c = np.delete(M, index, 0)
        if np.any(np.all(c > M[index], axis=1)):
            yield index


def strongly_dominated_columns(matrix):
    #generator version
    M = matrix.copy()
    for index in range(M.shape[1]):
        c = np.delete(M, index, 1)
        if np.any(np.all((c.transpose() < M[:, index]).transpose(), axis=0)):
            yield index


def pruning_dominated(matrix):
    M=matrix.copy()
    dominated_row = strongly_dominated_row(matrix)
    dominated_column = strongly_dominated_column(matrix)
    while dominated_column is not None or dominated_row is not None:
        if dominated_row is not None: M = np.delete(M, dominated_row, 0)
        if dominated_column is not None: M = np.delete(M, dominated_column, 1)
        dominated_row = strongly_dominated_row(M)
        dominated_column = strongly_dominated_column(M)
    return M



