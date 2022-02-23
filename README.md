# conjugate-gradient
Some simple examples for showing how does conjugate gradient method work on the system of linear equations.

## Python Code

### Conjugate Gradient Algorithm

```python
import numpy as np

def conjugate_grad(A, b, maxiter = 5):
    n = A.shape[0]
    x = np.zeros(n)
    r = b - A @ x
    p = r.copy()
    r_old = np.inner(r, r)
    for it in range(maxiter):
        alpha = r_old / np.inner(p, A @ p)
        x += alpha * p
        r -= alpha * A @ p
        r_new = np.inner(r, r)
        if np.sqrt(r_new) < 1e-10:
            break
        beta = r_new / r_old
        p = r + beta * p
        r_old = r_new.copy()
    return x
```

### Solving the System of Linear Equations Ax = b

```python
import numpy as np

A = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])
x = conjugate_grad(A, b)
```

### Solving Lyapunov Equation

```python
import numpy as np

def conjugate_grad_Ly(A, W, maxiter = 5):
    n = A.shape[0]
    X = np.zeros((n, n))
    r = np.reshape(W - A @ X - X @ A.T, -1, order = 'F')
    p = r.copy()
    r_old = np.inner(r, r)
    for it in range(maxiter):
        Q = p.reshape([n, n], order = 'F')
        alpha = r_old / np.inner(p, np.reshape(A @ Q + Q @ A.T, -1, order = 'F'))
        X += alpha * Q
        r -= alpha * np.reshape(A @ Q + Q @ A.T, -1, order = 'F')
        r_new = np.inner(r, r)
        if np.sqrt(r_new) < 1e-10:
            break
        beta = r_new / r_old
        p = r + beta * p
        r_old = r_new.copy()
    return X
```

```python
import numpy as np

A = np.array([[-2, -1], [-1, 0]])
W = np.array([[-1, 0], [0, -1]])
X = conjugate_grad_Ly(A, W)
```

### Solving Sylvester Equation

```python
import numpy as np

def compute_Ax(A, B, X):
    return np.reshape(A.T @ A @ X + A.T @ X @ B + A @ X @ B.T + X @ B @ B.T, -1, order = 'F')

def conjugate_grad_Sy(A, B, C, maxiter = 5):
    dim1 = A.shape[1]
    dim2 = B.shape[0]
    X = np.random.randn(dim1, dim2)
    x = np.reshape(X, -1, order = 'F')
    b = np.reshape(A.T @ C + C @ B.T, -1, order = 'F')
    Ax = compute_Ax(A, B, X)
    r = b - Ax
    p = r.copy()
    r_old = np.inner(r, r)
    for it in range(maxiter):
        Ap = compute_Ax(A, B, np.reshape(p, (dim1, dim2), order = 'F'))
        alpha = r_old / np.inner(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        r_new = np.inner(r, r)
        if np.sqrt(r_new) < 1e-10:
            break
        p = r + (r_new / r_old) * p
        r_old = r_new.copy()
    return np.reshape(x, (dim1, dim2), order = 'F')
```

```python
import numpy as np

A = np.array([[1, 0, 2, 3], [4, 1, 0, 2], [0, 5, 5, 6], [1, 7, 9, 0]])
B = np.array([[0, -1], [1, 0]])
C = np.array([[1, 0], [2, 0], [0, 3], [1, 1]])
X = conjugate_grad_Sy(A, B, C)
```
