from _collections_abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# falcon yards per play / points per game
coordinates = np.array([[5, 6, 5, 5, 6, 5, 6, 5], [13, 3, 35, 45, 23, 23, 29, 0]])

X = 0
Y = 1


def two_d_array_check(xycords: np.ndarray) -> np.ndarray:
    if xycords.ndim != 2:
        raise TypeError("xy cords must be a two tuple!")
    else:
        return xycords


two_d_array_check(coordinates)


def mean_func(xy_coords) -> np.ndarray:
    arr_x = xy_coords[0]
    arr_y = xy_coords[1]
    if arr_x.size > 0 and arr_y.size > 0:
        mean_x = arr_x.mean()
        mean_y = arr_y.mean()

        meanArr: np.ndarray = np.array([mean_x, mean_y])

        return meanArr
    else:
        print("no arrays \n")


meanCoords = mean_func(coordinates)


def height_at_mean(arr: np.ndarray, x: int) -> float | int:
    mean_height = 0
    if arr.size > 0:
        slope = arr[1] / arr[0]
        mean_height = slope * (x - arr[0]) + arr[1]
    return mean_height


def residuals(arr_xy) -> np.ndarray:
    arr_x = arr_xy[0]
    arr_y = arr_xy[1]
    r1 = arr_y - height_at_mean(meanCoords, arr_x)

    return r1


def residual_sum_is_zero(res_arr: np.ndarray) -> int | float:
    total: int = np.sum(res_arr)

    if np.isclose(total, 0):
        return total
    else:
        raise ValueError("sum of residuals not equal to zero. wrong line of best fit.")


def ssr_derivative_formula(arr_dx: np.ndarray, arr_dy: np.ndarray) -> dict:
    dx_dy = arr_dx * arr_dy
    dx_squared = arr_dx**2
    return {"dx_dy": dx_dy, "dx_squared": dx_squared}


type regSlope = float | int


def ssrm_derivative(
    is_zero_check: Callable[[np.ndarray], Any], res_func: Callable[[np.ndarray], Any]
) -> regSlope:

    minima = 0

    if np.isclose(is_zero_check(res_func(coordinates)), 0):
        dx = coordinates[X] - meanCoords[X]
        dy = coordinates[Y] - meanCoords[Y]

    if dx.any() and dy.any():
        products = ssr_derivative_formula(dx, dy)
        delta_y = np.sum(products["dx_dy"])
        delta_x = np.sum(products["dx_squared"])
        try:
            minima = delta_y / delta_x
        except ZeroDivisionError as e:
            print(f"error is: {e}")

        return minima

    else:
        print("failed to find derivative\n")


regression_slope = ssrm_derivative(residual_sum_is_zero, residuals)


def graphical_plot(xy_coords: np.ndarray, regression_slope: int) -> None:
    """
    Display a scatter plot of coordinate data and overlay a green regression
    line. The plot is rendered with Matplotlib and shown immediately.

    :param xy_coords: Coordinate data arranged so that the first row contains
        x-values and the second row contains y-values.
    :param regression_slope: Slope used for the overlaid regression line.
    :return: None.
    """
    plt.scatter(xy_coords[0], xy_coords[1])
    plt.axline((meanCoords[X], meanCoords[Y]), slope=regression_slope, color="green")
    plt.show()


def R2() -> None:

    predicted_y = (regression_slope * (coordinates[X] - meanCoords[X])) + meanCoords[Y]

    regression_margin = coordinates[Y] - predicted_y

    mean_margin = coordinates[Y] - meanCoords[Y]

    regression_margin = np.sum(np.square(regression_margin))

    mean_margin = np.sum(np.square(mean_margin))

    r2 = 1 - (regression_margin / mean_margin)

    print(f"the r^2 weight is : {r2}")


R2()

graphical_plot(coordinates, regression_slope)

# TODO clean up and refactor to better design with clearer function names


# TODO convert into console application (optional)
