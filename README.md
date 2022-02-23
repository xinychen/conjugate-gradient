# conjugate-gradient
Some simple examples for showing how does conjugate gradient method work on the system of linear equations.

## Python Code

```python
import numpy as np

def conjugate_grad(A, b, maxiter = 5):
    n = A.shape[0]
    x = np.zeros(n)
    r = A @ x - b
    p = r.copy()
    r_old = np.inner(r, r)
    for it in range(maxiter):
        alpha = r_old / np.inner(q, A @ p)
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
