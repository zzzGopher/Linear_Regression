from typing import Any, Callable

import numpy as np
from numpy import dtype, ndarray


myArrX = np.array([1,2,3])
myArrY = np.array([2,2,5])

def meanFunc(arrX:np.ndarray,arrY:np.ndarray)-> ndarray[tuple[Any, ...], dtype[Any]]:
    if (arrX.size > 0 and arrY.size > 0):
        meanX = arrX.mean()
        meanY = arrY.mean()

        meanArr:np.ndarray = np.array([meanX,meanY])

        return meanArr
    else:
        print("no arrays \n")

def lineHeight(arr:np.ndarray,x:int)->int:
    line_height = 0
    if arr.size > 0:
        slope = (arr[0]/arr[1])
        line_height = (slope * (x - arr[0]) + arr[1])
    return line_height

def residuals(arrY:np.ndarray,arrX:np.ndarray)->np.ndarray:
    r1 = np.array([])

    for i in range(len(arrY)):
        res1 = arrY[i] - lineHeight(meanFunc(myArrX,myArrY),arrX[i])
        r1 = np.append(r1,res1)

    print(f"residuals: {r1} \n")
    return r1

def srCheck(res_arr:np.ndarray)->int:
    total:int = 0
    for i in range(len(res_arr)):
        total += res_arr[i]

    if total == 0:
        return total
    else:
        raise ValueError("sum of residuals not equal to zero. wrong line of best fit.")

# at this point we have the residuals check and need to get the SSR(m) and find the derivative at zero to form the regression line

type callback = Callable[...,int]
type callback2 = Callable[[np.ndarray,np.ndarray],np.ndarray]

def mult_dots(arr_dx:np.ndarray,arr_dy:np.ndarray)->dict:
    dx_dy = np.array([])
    dx_squared = np.array([])
    for i in range(len(arr_dx)):
        dx_dy = np.append(dx_dy,(arr_dx[i] * arr_dy[i]))
        dx_squared = np.append(dx_squared,(arr_dx[i]**2))
    return {'dx_dy':dx_dy,'dx_squared':dx_squared}


def ssrm_derivative(sr_check:callback,res_func:callback2)->int|None:
    x = 0
    y = 1
    dx = np.array([])
    dy = np.array([])
    minima = 0

    if sr_check(res_func(myArrY,myArrX)) == 0:
        mean = meanFunc(myArrX,myArrY)

        for i in range(len(myArrX)):
            dx = np.append(dx,(myArrX[i] - mean[x]))
            dy = np.append(dy,(myArrY[i] - mean[y]))

#TODO fix hardcoded array length check
    if len(dx) == 3 and len(dy) == 3:
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


print(ssrm_derivative(srCheck,residuals))