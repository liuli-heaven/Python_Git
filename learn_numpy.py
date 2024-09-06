import numpy as np


def test1():
    array = np.array([
        [1, 3, 5],
        [2, 4, 6]
    ])

    print(array)

    # 1.2 维度
    print('number of dim:', array.ndim);

    # 1.3 行数和列数
    print(f"shape:{array.shape}")

    # 1.4 元素个数
    print(f"size:{array.size}")

def test2():
    a = np.array([2, 23, 4], dtype=np.int32())
    print("a:", a, " a.dtype:", a.dtype)

    # 2.2 多维array创建
    a = np.array([
        [2, 3, 4],
        [5, 6, 7]
    ])
    print(a)

    # 2.3 创建全零数组
    a = np.zeros((2, 3), dtype=np.int32())
    print(a)

    # 2.4 创建全1数组
    a = np.ones((2, 3))
    print(a)

    # 2.5 创建全空数组
    a = np.empty((2, 3))
    print(a)

    # 2.6 创建连续数组
    a = np.arange(10, 21, 2)
    print(a)

    # 2.7 reshape操作
    a = np.random.rand(6).reshape((2, 3))
    print(a)

    # 2.8 创建连续型数据
    a = np.linspace(1, 10, 20)
    print(a)

    # 2.9 linspace的reshape操作
    b = a.reshape((5, 4))
    print(b)

def test3():
    a = np.array([10, 20, 30, 40])
    b = np.arange(4)
    print(a, b)

    print(a - b)

    print(a * b)

    print(b ** 2)

    c = np.sin(a)
    print(c)

    print(b < 2)

    a = np.array([1, 1, 4, 3])
    b = np.arange(4)
    print(a == b)

    #3.2 多维矩阵运算
    a = np.array([[1, 1], [0, 1]])
    b = np.arange(4).reshape((2, 2))
    print(a)

    print(b)

    c = a.dot(b) #  乘法
    print(c)

    c = np.dot(a, b)
    print(c)

    a = np.random.random((2, 4))
    print(np.sum(a))

    print(np.min(a))
    print(np.max(a))

    print(f"a={a}")

    # 对sum函数来说，axis为0时，将第0维度上的n个数据合并
    print(f"sum={np.sum(a, axis=1)}")

    # 对min函数来说，axis为0时，对第0维度上的n个数据分别取最小值。
    print("min=", np.min(a, axis=0))

    # 3.3 基本计算

    A = np.arange(2, 14).reshape((3, 4))
    print(A)

    print(np.argmin(A))

    print(np.argmax(A))

    print(np.mean(A))

    print(np.average(A))

    print(A.mean())

    print(np.median(A))

    print(np.cumsum(A))

    B = np.array([[3, 5, 9],
                  [4, 8, 10]])
    print(np.diff(B))

    C = np.array([[0, 5, 9],
                  [4, 0, 10]])
    print(np.nonzero(B))
    print(np.nonzero(C))

    A = np.arange(14, 2, -1).reshape((3, 4))
    print(A)

    print(np.sort(A))

    print(np.transpose(A))

    print(A.T)

    print(A)

    print(np.clip(A, 5, 9))

# numpy索引和切片
def test4():
    A = np.arange(3, 15)
    print(A)
    print(A[3])

    B = A.reshape(3, 4)
    print(B)

    print(B[2])
    print(B[0, 2])
    print(B[1, 1:3])

    for val in B:
        print(val)

    for column in B.T:
        print(column)

    A = np.arange(3, 15).reshape((3, 4))
    print(A.flatten())


    for item in A.flat:
        print(item)

#numpy array合并
def test5():
    A = np.array([1, 1, 1])
    B = np.array([2, 2, 2])
    print(np.vstack((A, B)))

    C = np.vstack((A, B))
    print(C)

    print(A.shape, B.shape, C.shape)
    D = np.hstack((A, B))
    print(D)

    print(A[np.newaxis, :])
    print(A[np.newaxis, :].shape)
    print(A[:,np.newaxis])

    A = A[:, np.newaxis]
    B = B[np.newaxis]
    C = np.concatenate((A, B, B, A), axis=0)
    print(C)

    C = np.concatenate((A, B), axis=1)
    print(C)

def test6():
    A = np.arange(12).reshape((3,4))
    print(A)

    print(np.split(A, 2, axis=1))
    print(np.split(A, 3, axis=0))

    print(np.array_split(A, 3, axis=1))

    print(np.vsplit(A, 3))
    print(np.hstack(A, 2))

def test7():
    a = np.arange(4)
    print(a)
    b = a
    c = a
    d = b
    a[0] = 11
    print(a, b, c, d)
    """说明=相当于进行了浅拷贝，实际上此处py中的=是在进行引用"""

    a = np.arange(4)
    print(a)
    b = a.copy()
    print(b)
    a[3] = 44
    print(a)
    print(b)

# 广播机制
def test8():
    a = np.array([[0, 0, 0],
               [10, 10, 10],
               [20, 20, 20],
               [30, 30, 30]])
    b = np.array([0, 1, 2])
    print(a + b)
    b = np.tile([0,1,2],(4,1))
    print(a+b)

def test9():
    x = np.array([1, 2, 3, 3, 0, 1, 4])
    np.bincount(x)
    w = np.array([0.3, 0.5, 0.7, 0.6, 0.1, -0.9, 1])
    np.bincount(x, weights=w)
    np.bincount(x, weights=w, minlength=7)

    x = [[1, 3, 3],
         [7, 5, 2]]
    print(np.argmax(x))

    np.around([-0.6, 1.2798, 2.357, 9.67, 13], decimals=0)
    np.around([1.2798, 2.357, 9.67, 13], decimals=1)
    np.around([1.2798, 2.357, 9.67, 13], decimals=2)
    np.around([1, 2, 5, 6, 56], decimals=-1)

    # 计算沿指定轴第N维的离散差值
    x = np.arange(1, 16).reshape((3, 5))
    print(x)
    np.diff(x, axis=1)  # 默认axis为1
    np.diff(x, axis=0)

    np.floor([-0.6,-1.4,-0.1,-1.8,0,1.4,1.7])  # 向下取整
    np.ceil([-0.6,-1.4,-0.1,-1.8,0,1.4,1.7])  # 向上取整

    x = np.array([[1, 0],
                  [2, -2],
                  [-2, 1]])
    print(x)

    np.where(x>0, x, 0)
