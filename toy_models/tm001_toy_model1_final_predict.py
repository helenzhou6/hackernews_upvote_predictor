from scipy.optimize import fsolve
import numpy as np
import pandas as pd

# # Fit a degree-3 polynomial in log(x)
# t = np.log(x_true)
# coeffs = np.polyfit(t, y_true, 3)

# # Fit a degree-3 polynomial in log(x)
# t = np.log(x_true)
# coeffs = np.polyfit(t, y_true, 3)

coeffs = np.array([0.07538848, -0.76435324,  3.6821385 , -0.24469317])
y_true = pd.read_csv()
print(y_true)


# Define the polynomial function
def poly_logx(x, y_target):
    logx = np.log(x)
    y = np.polyval(coeffs, logx)
    return y - y_target  # We want this to be 0

# Solve for x given y, element-wise
x_recovered = []
for y_val in y_true:
    # Provide an initial guess (you can tweak this)
    x_guess = 1.0
    x_sol = fsolve(poly_logx, x0=x_guess, args=(y_val,))
    x_recovered.append(x_sol[0])

x_recovered = np.round(x_recovered, 0)

# compute loss
mse = np.mean((x_true - x_recovered)**2)
print(mse)
