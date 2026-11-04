from typing import Any, Callable

import numpy as np
from fontTools.diff import color
from numpy import dtype, ndarray
import matplotlib.pyplot as plt


coordinates = np.array([[1,2,3],[2,2,5]])

def data_points(xycords:np.ndarray)->np.ndarray:
    if xycords.ndim != 2:
        raise TypeError("xy cords must be a two tuple!")
    else:
        return xycords
    

def meanFunc(xy_coords)-> np.ndarray:
    arr_x = xy_coords[0]
    arr_y = xy_coords[1]
    if (arr_x.size > 0 and arr_y.size > 0):
        mean_x = arr_x.mean()
        mean_y = arr_y.mean()

        meanArr:np.ndarray = np.array([mean_x,mean_y])

        return meanArr
    else:
        print("no arrays \n")

def lineHeight(arr:np.ndarray,x:int)->int:
    line_height = 0
    if arr.size > 0:
        slope = (arr[1]/arr[0])
        line_height = (slope * (x - arr[0]) + arr[1])
    return line_height

def residuals(arr_xy)->np.ndarray:
    r1 = np.array([])
    arr_x = arr_xy[0]
    arr_y = arr_xy[1]

    for i in range(len(arr_y)):
        res1 = arr_y[i] - lineHeight(meanFunc(coordinates),arr_x[i])
        r1 = np.append(r1,res1)

    return r1

def sr_is_zero_check(res_arr:np.ndarray)->int:
    total:int = 0
    for i in range(len(res_arr)):
        total += res_arr[i]

    if np.isclose(total,0):
        return total
    else:
        raise ValueError("sum of residuals not equal to zero. wrong line of best fit.")

# at this point we have the residuals check and need to get the SSR(m) and find the derivative at zero to form the regression line

type callback = Callable[...,np.ndarray]
type callback2 = Callable[[np.ndarray],np.ndarray]

def mult_dots(arr_dx:np.ndarray,arr_dy:np.ndarray)->dict:
    dx_dy = np.array([])
    dx_squared = np.array([])
    for i in range(len(arr_dx)):
        dx_dy = np.append(dx_dy,(arr_dx[i] * arr_dy[i]))
        dx_squared = np.append(dx_squared,(arr_dx[i]**2))
    return {'dx_dy':dx_dy,'dx_squared':dx_squared}


def ssrm_derivative(is_zero_check:callback,res_func:callback2)->int|None:
    x = 0
    y = 1
    dx = np.array([])
    dy = np.array([])
    minima = 0

    if np.isclose(is_zero_check(res_func(data_points(coordinates))), 0):
        x_coords = coordinates[0]
        y_coords = coordinates[1]
        mean = meanFunc(coordinates)

        for i in range(len(x_coords)):
            dx = np.append(dx,(x_coords[i] - mean[x]))
            dy = np.append(dy,(y_coords[i] - mean[y]))


    if dx.any() and dy.any() :
        products = mult_dots(dx,dy)
        delta_y = np.sum(products['dx_dy'])
        delta_x = np.sum(products['dx_squared'])
        try:
            minima = (delta_y/delta_x)
        except ZeroDivisionError as e:
            print(f"error is: {e}")

        return minima

    else:
        print("failed to find derivative\n")


def data_plot(xy_coords:np.ndarray,regression_slope:int,mean:callback):
    mean = meanFunc(xy_coords)
    plt.scatter(xy_coords[0],xy_coords[1])
    plt.axline((mean[0],mean[1]),slope=regression_slope,color="green")
    plt.show()
    return regression_slope


regression_slope = ssrm_derivative(sr_is_zero_check,residuals)

def sum_squared_res(res):
    sum = 0
    for i in range(len(res)):
        sum += i
    if sum > 0:
        return sum

def R2():
    reg_slope = data_plot(coordinates, regression_slope, meanFunc(coordinates))

    regression_margin = np.array([])

    y_coords = coordinates[1]
    x_coords = coordinates[0]
    predicted_y = np.array([])

    mean_coords = meanFunc(coordinates)
    mean_coordx = mean_coords[0]
    mean_coordy = mean_coords[1]

    ssr_mean = np.array([])


    for i in range(len(x_coords)):
        y = (reg_slope * (x_coords[i] - mean_coordx)) + mean_coordy
        predicted_y = np.append(predicted_y,y)


    for i in range(len(coordinates[0])):
        margin1 = predicted_y[i] - y_coords[i]
        regression_margin = np.append(regression_margin,margin1)

    for i in range(len(y_coords)):
        res = y_coords[i] - mean_coordy
        ssr_mean = np.append(ssr_mean,res)


    ssr_regression = np.sum(np.square(regression_margin))

    ssr_mean = np.sum(np.square(ssr_mean))

    print(ssr_regression,ssr_mean)

    r2 = 1 - (ssr_regression/ssr_mean)

    print(f"the r^2 weight is : {r2}")


R2()

#TODO clean up and refactor to better design with clearer function names


#TODO convert into console application (optional)

