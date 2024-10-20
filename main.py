import math

def fibo_1(n):
    if n <= 1:
        return n

    return fibo_1(n-1) + fibo_1(n-2)

# print(fibo_1(23))

def fibo_memo(n):
    memo = [-1] * (n+1)
    if n <= 1:
        return n

    if memo[n] != -1:
        return memo[n]

    memo[n] = fibo_memo(n-1) + fibo_memo(n-2)
    return memo[n]

# print(fibo_memo(23))

def fibo_dp(n):
    dp = [0] * (n+1)

    if n <= 1:
        return n

    dp[0] = 0
    dp[1] = 1

    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]

# print(fibo_dp(323))

def fibo_dp_2(n):
    if n <= 1:
        return n
    curr = 0

    prev1 = 1
    prev2 = 0
    for i in range(2, n+1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr

    return curr

# print(fibo_dp_2(9999))

def multiply(mat1, mat2):
    x = mat1[0][0] * mat2[0][0] + mat1[0][1] * mat2[1][0]
    y = mat1[0][0] * mat2[0][1] + mat1[0][1] * mat2[1][1]
    z = mat1[1][0] * mat2[0][0] + mat1[1][1] * mat2[1][0]
    t = mat1[1][0] * mat2[0][1] + mat1[1][1] * mat2[1][1]

    mat1[0][0], mat1[0][1] = x, y
    mat1[1][0], mat1[1][1] = z, t

def matrix_power(mat1, n):
    if n == 0 or n == 1:
        return

    mat2 = [[1, 1], [1, 0]]

    matrix_power(mat1, n // 2)

    # Square the matrix mat1
    multiply(mat1, mat1)

    # If n is odd, multiply by the helper matrix mat2
    if n % 2 != 0:
        multiply(mat1, mat2)

def fibo_matrix(n):
    if n <= 1:
        return n

    # Initialize the transformation matrix
    mat1 = [[1, 1], [1, 0]]

    # Raise the matrix mat1 to the power of (n - 1)
    matrix_power(mat1, n - 1)

    # The result is in the top-left cell of the matrix
    return mat1[0][0]

print(fibo_matrix(10000))