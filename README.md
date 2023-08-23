# Code_Plateau_DS_Journey
This is my journey of learning Data Science at Code Plateau

This is an exercise after learning about numpy
Below are the questions and how i tried to solve them
Some has more than one approaches used


___
___

# NumPy Exercises 

Now that we've learned about NumPy let's test your knowledge. We'll start off with a few simple tasks, and then you'll be asked some more complicated questions.

#### Import NumPy as np


```python
#First We Need To Import Numpy
import numpy as np
```

#### Create an array of 10 zeros 


```python
#Solution and The Output is Below
np.zeros(10)
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])



#### Create an array of 10 ones


```python
#Solution and The Output is Below
np.ones(10)
```




    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])



#### Create an array of 10 fives


```python
#Solution and The Output is Below
a = np.ones(10)
a *5
```




    array([5., 5., 5., 5., 5., 5., 5., 5., 5., 5.])



#### Create an array of the integers from 10 to 50


```python
#Solution and The Output is Below
b = np.arange(10,51)
b
```




    array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
           27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
           44, 45, 46, 47, 48, 49, 50])



#### Create an array of all the even integers from 10 to 50


```python
#Solution and The Output is Below
np.arange(10,51,2)
```




    array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42,
           44, 46, 48, 50])



#### Create an array of all the odd integers from 10 to 50


```python
#Solution and The Output is Below
np.arange(11,51,2)
```




    array([11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43,
           45, 47, 49])



#### Create a 3x3 matrix with values ranging from 0 to 8


```python
#Method One
#Solution and The Output is Below
np.array(([0,1,2],[3,4,5],[6,7,8]))
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
#Method Two
#Solution and The Output is Below
c = np.arange(0,9)
d=c.reshape(3,3)
d
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])



#### Create a 3x3 identity matrix


```python
#Solution and The Output is Below
np.eye(3,3)
```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])



#### Use NumPy to generate a random number between 0 and 1


```python
#Solution and The Output is Below
np.random.rand(1)
```




    array([0.18597398])



#### Use NumPy to generate an array of 25 random numbers sampled from a standard normal distribution


```python
#Solution and The Output is Below
np.random.randn(25)
```




    array([ 0.94084113,  0.60325334,  0.37502661,  0.74672215, -0.03460223,
            0.03168189,  1.08200881, -0.05731399,  2.03714932, -0.98001347,
           -0.40950561, -0.32418837,  1.09626695,  0.22959759,  0.34401352,
           -1.70230749, -0.31214916, -1.64131078, -0.77826634,  0.72198578,
           -1.15801953, -0.48704746, -1.02955775, -1.10225432, -0.45328278])




```python
#Solution and The Output is Below
np.random.normal(0,1,25)
```




    array([-0.70392411, -0.71374706,  0.08992806, -0.01102353, -0.25204433,
            1.66494943,  0.19131414,  1.96455056,  0.3277305 ,  0.45821123,
           -0.43501549,  0.92862085, -0.57333215, -0.3564923 , -0.53980323,
            0.45281664,  0.8941829 ,  1.46527733, -1.45365794,  2.46314016,
            0.00416058,  0.0437772 , -0.17907912,  0.51011909,  1.79136421])



#### Create the following matrix:


```python
#Solution and The Output is Below
f = np.arange(0.01,1.01,0.01)
e = f.reshape(10,10)
print(e)
```






    [[0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 ]
     [0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 ]
     [0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 ]
     [0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 ]
     [0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 ]
     [0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 ]
     [0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 ]
     [0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 ]
     [0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 ]
     [0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.  ]]
    



#### Create an array of 20 linearly spaced points between 0 and 1:


```python
#Solution and The Output is Below
np.linspace(0, 1, 20)
```




    array([ 0.        ,  0.05263158,  0.10526316,  0.15789474,  0.21052632,
            0.26315789,  0.31578947,  0.36842105,  0.42105263,  0.47368421,
            0.52631579,  0.57894737,  0.63157895,  0.68421053,  0.73684211,
            0.78947368,  0.84210526,  0.89473684,  0.94736842,  1.        ])



## Numpy Indexing and Selection

Now you will be given a few matrices, and be asked to replicate the resulting matrix outputs:


```python
#Solution and The Output is Below
mat = np.arange(1,26).reshape(5,5)
mat
```




    array([[ 1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10],
           [11, 12, 13, 14, 15],
           [16, 17, 18, 19, 20],
           [21, 22, 23, 24, 25]])




```python
# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE
```


```python

```


```python
# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE
```


```python
#Solution and The Output is Below
#Using the above matrix to output the item '20'
mat[3,4]
```




    20




```python
# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE
```

```python
#Solution and The Output is Below
h = np.arange(1,16)
g = h.reshape(3,5)
f = g[0:,1]
for i in f:
    print(i)
```

    2
    7
    12
    


```python
# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE
```

```python
#Solution and The Output is Below
ary = np.arange(21,26)
ary
```




    array([21, 22, 23, 24, 25])




```python
# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE
```

```python
#Solution and The Output is Below
are = np.arange(16,26)
ar = are.reshape(2,5)
ar
```




    array([[16, 17, 18, 19, 20],
           [21, 22, 23, 24, 25]])





### Now do the following

#### Get the sum of all the values in mat


```python
#Solution and The Output is Below
print(np.sum(mat))
```

    325
    

#### Get the standard deviation of the values in mat


```python
#Solution and The Output is Below
print(np.std(mat))
```

    7.211102550927978
    

#### Get the sum of all the columns in mat


```python
#Solution and The Output is Below
print(np.sum(mat, axis=0))
```

    [55 60 65 70 75]
    

# Great Job!
