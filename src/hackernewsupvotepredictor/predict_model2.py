# from scipy.optimize import fsolve
# import numpy as np
# import pandas as pd
import torch

# # # Fit a degree-3 polynomial in log(x)
# # t = np.log(x_true)
# # coeffs = np.polyfit(t, y_true, 3)

# # # Fit a degree-3 polynomial in log(x)
# # t = np.log(x_true)
# # coeffs = np.polyfit(t, y_true, 3)

# # def pred_final_score():

# # def get_final_pred(y_true):

# coeffs = torch.tensor([0.07538848, -0.76435324,  3.6821385 , -0.24469317])
# y_true = torch.load("data/target.pt")

# # y_true = pd.read_csv("data/")

# # Define the polynomial function
# def poly_logx(x, y_target):
#     logx = torch.log(x)
#     y = tf.math.polyval(coeffs, logx)
#     # y = np.polyval(coeffs, logx)
#     return y - y_target  # We want this to be 0

# # Solve for x given y, element-wise
# x_recovered = []
# for y_val in y_true:
#     # Provide an initial guess (you can tweak this)
#     x_guess = 1.0
#     x_sol = fsolve(poly_logx, x0=x_guess, args=(y_val,))
#     x_recovered.append(x_sol[0])

# x_recovered = np.round(x_recovered, 0)
# x_recovered = torch.tensor(x_recovered)
# x_recovered.requires_grad = False
#     # return x_recovered

#     # # compute loss
#     # mse = np.mean((x_true - x_recovered)**2)
#     # print(f"mse is {mse}")

# Constant polynomial coefficients (highest degree first)
coeffs = torch.tensor([0.07538848, -0.76435324,  3.6821385 , -0.24469317])

def polyval(coeffs, x):
    """Evaluate a polynomial at x with torch ops."""
    y = torch.zeros_like(x)
    for i, c in enumerate(coeffs):
        y += c * x ** (len(coeffs) - i - 1)
    return y

def poly_logx(x, y_target, coeffs):
    logx = torch.log(x)
    y = polyval(coeffs, logx)
    return y - y_target

def d_poly_logx_dx(x, coeffs):
    logx = torch.log(x)
    # Derivative of the polynomial with respect to logx
    d_coeffs = torch.tensor(
        [c * (len(coeffs) - i - 1) for i, c in enumerate(coeffs[:-1])],
        dtype=torch.float32, device=x.device
    )
    dy_dlogx = polyval(d_coeffs, logx)
    return (1 / x) * dy_dlogx  # dy/dx = dy/dlogx * dlogx/dx

def get_final_pred(y_target, coeffs=torch.tensor([0.07538848, -0.76435324,  3.6821385 , -0.24469317]), max_iter=20, tol=1e-6):
    # y_target: Tensor of shape [N]
    x = torch.ones_like(y_target, requires_grad=True)

    for _ in range(max_iter):
        fx = poly_logx(x, y_target, coeffs)
        dfx = d_poly_logx_dx(x, coeffs)
        step = fx / dfx
        x = x - step
        if torch.max(torch.abs(step)) < tol:
            break
    return x


