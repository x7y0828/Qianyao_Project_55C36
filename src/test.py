import numpy as np
from src.operators import grad, div

##Test for Step 2###
# Test 1: Gradient of a constant image
constant = np.ones((5, 5))
a, b = grad(constant)
print("Constant gradient:", np.max(np.abs(a)), np.max(np.abs(b)))  # should be 0

assert np.allclose(a,0)
assert np.allclose(b,0)

# Test 2: Negative adjoint relationship
rng = np.random.default_rng(42)

u = rng.normal(size=(5, 5))
p1 = rng.normal(size=(5, 5))
p2 = rng.normal(size=(5, 5))

gu1, gu2 = grad(u)

lhs = np.sum(gu1*p1 + gu2*p2)
rhs = -np.sum(u * div(p1, p2))

print("Left side:", lhs)
print("Right side:", rhs)
print("Difference:", np.abs(lhs - rhs))

assert np.allclose(lhs, rhs)

print("All operator tests passed!")