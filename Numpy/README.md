# Numpy 教程

***

## 一、先决条件

***

在阅读本教程之前，你应该了解一下Python的使用方法。如果你想复习一下Python，请查看[Python教程](https://docs.python.org/3/tutorial/)

如果你希望使用本教程中的实例，则还必须在计算机上安装某些软件。有关说明请参阅[此网站](https://scipy.org/install.html)

## 二、基础

***

Numpy的主要对象是同构多维数组，它是一个元素的列表（通常是数字），其中元素都是相同的类型，有正整数元组索引。在Numpy中，数组的维度称为轴。

例如，3D空间中点的坐标[1,2,1]具有一个轴。该轴有3个元素，所以我们说它的长度为3。在下图所示的例子中，数组有2个轴。第一轴的长度为2，第二轴的长度为3。

> [ [1，0，0]，
>   [0，1，2] ]

NumPy中的数组类称为**ndarray**。它也被别名数组所知。请注意，numpy.array与标准Python库类array.array不同，后者仅处理一维数组并提供较少的功能。ndarray对象的具有一些更重要的属性是：

1. ndarray.ndim
阵列的轴数（即数组的维度数量）

2. ndarray.shape
数组的大小。这是一个整数元组，其中的元素表示每个维度中数组的长度。对于具有n行和m列的矩阵，ndarray.shape将表示为（n，m）。因此，ndarray.shape的长度是数组的轴的数量ndim。

3. ndarray.size
数组的元素总数。这等于ndarray.shape元素的乘积。

4. ndarray.dtype
描述数组中元素类型的对象。可以使用标准Python类型创建或指定dtype。此外，NumPy还提供自己的类型。 如numpy.int32，numpy.int16和numpy.float64等一些例子。

5. ndarray.itemsize
数组中每个元素的大小（以字节为单位）。例如，float64类型的元素数组具有itemsize 8（= 64/8），而complex32类型之一具有itemsize 4（= 32/8）。它相当于ndarray.dtype.itemsize。

6. ndarray.data
包含数组实际元素的缓冲区。通常，我们不需要使用此属性，因为我们将使用索引工具访问数组中的元素。

### 示例

```python
>>> import numpy as np
>>> a = np.arange(15).reshape(3, 5)
>>> a
array([[ 0,  1,  2,  3,  4],
[ 5,  6,  7,  8,  9],
[10, 11, 12, 13, 14]])
>>> a.shape
(3, 5)
>>> a.ndim
2
>>> a.dtype.name
'int64'
>>> a.itemsize
8
>>> a.size
15
>>> type(a)
<type 'numpy.ndarray'>
>>> b = np.array([6, 7, 8])
>>> b
array([6, 7, 8])
>>> type(b)
<type 'numpy.ndarray'>
```

### 数组的创建

这里有很多种方法可以创建数组。

例如，可以使用array函数从常规Python列表或元组创建数组。数组元素的类型是从根据列表或元组中元素的类型推导出来的。

```python
>>> import numpy as np
>>> a = np.array([2,3,4])
>>> a
array([2, 3, 4])
>>> a.dtype
dtype('int64')
>>> b = np.array([1.2, 3.5, 5.1])
>>> b.dtype
dtype('float64')
```

经常出现的错误在于使用array函数调用多个数字作为函数参数，而不是提供数字列表作为参数。

```python
>>> a = np.array(1,2,3,4)    # WRONG
>>> a = np.array([1,2,3,4])  # RIGHT
```

array函数自动将内嵌列表的列表转换为二维数组，将内嵌双层列表的列表转换为三维数组，等等。

```python
>>> b = np.array([(1.5,2,3), (4,5,6)])
>>> b
array([[ 1.5,  2. ,  3. ],
[ 4. ,  5. ,  6. ]])
```

也可以在创建时直接指定数组的类型：

```python
>>> c = np.array( [ [1,2], [3,4] ], dtype=complex )
>>> c
array([[ 1.+0.j,  2.+0.j],
[ 3.+0.j,  4.+0.j]])
```

通常，数组的元素最初是未知的，但其大小是已知的。因此，NumPy提供了几个函数来创建具有初始占位符内容的数组。这些最小化了增长阵列的必要性，这是一项代价高的操作。

函数zeros创建一个充满零的数组，函数ones创建一个充满1的数组，函数empty创建一个数组，其初始元素是随机的，取决于内存的状态。默认情况下，创建的数组的dtype是float64。

```python
>>> np.zeros( (3,4) )
array([[ 0.,  0.,  0.,  0.],
[ 0.,  0.,  0.,  0.],
[ 0.,  0.,  0.,  0.]])
>>> np.ones( (2,3,4), dtype=np.int16 )   # dtype can also be specified
array([[[ 1, 1, 1, 1],
[ 1, 1, 1, 1],
[ 1, 1, 1, 1]],
[[ 1, 1, 1, 1],
[ 1, 1, 1, 1],
[ 1, 1, 1, 1]]], dtype=int16)
>>> np.empty( (2,3) )         # uninitialized, output may vary
array([[  3.73603959e-262,   6.02658058e-154,   6.55490914e-260],
[  5.30498948e-313,   3.14673309e-307,   1.00000000e+000]])
```

为了创建数字序列，NumPy提供了一个类似于函数range的方法，它返回的是数组从而代替了直接写入列表。

```python
>>> np.arange( 10, 30, 5 )
array([10, 15, 20, 25])
>>> np.arange( 0, 2, 0.3 )     # it accepts float arguments
array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])
```

当arange与浮点参数一起使用时，由于有限的浮点精度，通常无法准确的获得元素数。出于这个原因，通常最好使用函数linspace作为参数接收我们想要的元素数，而不是上面那种方式：

```python
>>> from numpy import pi
>>> np.linspace( 0, 2, 9 )        # 9 numbers from 0 to 2
array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
>>> x = np.linspace( 0, 2*pi, 100 )   # useful to evaluate function at lots of points
>>> f = np.sin(x)
```

### 打印阵列

当需要打印数组时，NumPy以与嵌套列表类似的方式显示它，但具有以下布局：

最后一个轴从左到右打印，
倒数第二个从上到下打印，
其余部分也从上到下打印，每个切片用空行分隔。
然后将一维数组打印为行，将二维数据打印为矩阵，将三维数据打印为矩阵列表。

```python
>>> a = np.arange(6)                 # 1d array
>>> print(a)
[0 1 2 3 4 5]
>>>
>>> b = np.arange(12).reshape(4,3)   # 2d array
>>> print(b)
[[ 0  1  2]
[ 3  4  5]
[ 6  7  8]
[ 9 10 11]]
>>>
>>> c = np.arange(24).reshape(2,3,4) # 3d array
>>> print(c)
[[[ 0  1  2  3]
[ 4  5  6  7]
[ 8  9 10 11]]
[[12 13 14 15]
[16 17 18 19]
[20 21 22 23]]]
```

请参阅下文以获取有关重塑的更多详细信息。

如果数组太大而无法打印，NumPy会自动跳过数组的中心部分并仅打印角落：

```python
>>> print(np.arange(10000))
[   0    1    2 ..., 9997 9998 9999]
>>>
>>> print(np.arange(10000).reshape(100,100))
[[   0    1    2 ...,   97   98   99]
[ 100  101  102 ...,  197  198  199]
[ 200  201  202 ...,  297  298  299]
...,
[9700 9701 9702 ..., 9797 9798 9799]
[9800 9801 9802 ..., 9897 9898 9899]
[9900 9901 9902 ..., 9997 9998 9999]]
```

要禁用此行为并强制NumPy打印整个阵列，可以使用set_printoptions更改打印选项。

```python
>>> np.set_printoptions(threshold=np.nan)
```
