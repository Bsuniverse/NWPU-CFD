------
<br/><br/><br/><br/><br/><br/><br/><br/>

<p align="center"><font size="8">计算流体力学基础<br/></br/>大作业说明文档</font></p>

<br/><br/><br/><br/><br/><br/><br/><br/><br/>

<p align="center">学 号: </p>

<p align="center">姓 名: </p>

<p align="center">Tel: </p>

<br/><br/><br/><br/><br/>

<p align="center">xxxx年xx月</p>

------

<div style="page-break-after: always;"></div>

[TOC]

<div style="page-break-after: always;"></div>

# 0. Introduction

本文档为计算流体力学基础大大作业的说明文档

- 使用语言：Python 3.7版本；
- 本作业中用到的额外库：numpy (用于矩阵运算)， pandas (读取csv文件用)，matplotlib （用于绘制3D图片）
- Python额外库安装方法`python -m pip install --user numpy matplotlib pandas`
- 可选库：Tensorflow(C语言级别的矩阵计算速度，为机器学习、科学计算优化，比numpy的矩阵运算快5倍，矩阵较大时还可使用GPU计算)，可取代numpy。
- CPU配置：Intel(R) Xeon(R) CPU E5-2637 v4 @ 3.50GHZ 3.50GHZ (2 处理器)，计算时间供参考。

本次大作业分为三个，其文件夹格式为：

1. O网格生成
   1. 输入：input.txt
   2. 运行文件：ogrid.py
   3. 输出：Result/Cp-x.dat, Result/flowfield.dat, Result/o-grid.dat
2. 一维Burgers方程数值求解
   1. 输入：input.txt
   2. 运行文件：burgers.py
   3. 输出：2D文件夹中的mu=0.0.dat, mu=0.01.dat, mu=0.05.dat
   4. 可选输出：3D文件夹中的三个三维图，mu=0.0.png, mu=0.01.png, mu=0.05.png
3. 准一维Laval流动
   1. 输入：input.txt
   2. 运行文件：laval.py
   3. 输出：Result/Cx=0.05, 0.2, 0.15.dat

# 1. O网格生成与数值求解

## 1.1. 输入文档说明

输入文档格式如下：

```csv
Points,Radius,iterlimit,errlimit,Velocity 
101,20,20000,1e-05,20.0
```

- Points表示翼型取点个数N，其中后缘点算作两个点；
- Radius表示远场边界条件半径r；
- iterlimit表示最大容许的迭代次数；
- errlimit表示最大误差限，即迭代前后误差小于errlimit时停止迭代；
- Velocity表示最大来流速度$V_\infty$。

## 1.2. 运行文档<u>ogrid.py</u>说明

文档定义两个类：

```python
class Ogrid():
    # 输入参数：迭代次数N, 半径radius, 迭代和误差限
    def __init__(self, N, radius, iterlimit, errlimit)
    # 初始化网格函数
    def initialGrid(self)
    # 网格迭代生成函数
    def gridGenerate(self)
    # 网格导出为dat文件函数
    def gridExport(self, x, y)
class GridSolver():
    # 输入参数：网格点坐标x, y, 自由来流速度Vf, 迭代和误差限
    def __init__(self, x, y, Vf, errlimit, iterlimit)
    # 流函数迭代求解器
    def phiSolver(self)
    # 流场参数求解器
    def flowSolver(self)
    # 流畅参数x, y, u, v, Cp导出函数
    def flowExport(self, x, y, u, v, Cp)
```

迭代中差分格式数学表达式
$$
u_i^j=u_{i+1}^j-u_{i-1}^j
$$
差分格式python代码实现

```python
u[:, 1:-1]=u[:, 2:] - u[:, 0:-2]
```

main函数实现

```python
if __name__ == '__main__':
	input = pd.read_csv('input.txt', sep=',', usecols=['Points', 'Radius', \
        'iterlimit','errlimit', 'Velocity'])
	N, r, iterlimit, errlimit, Vf = input['Points'][0], input['Radius'][0], \
    input['iterlimit'][0], input['errlimit'][0], input['Velocity'][0]

	ogrid = Ogrid(N, r, iterlimit, errlimit)
	x, y = ogrid.gridGenerate()
	ogrid.gridExport(x, y)
	flow = GridSolver(x, y, Vf, errlimit, iterlimit)
	u, v, Cp, phi = flow.flowSolver()
	flow.flowExport(x, y, u, v, Cp)
```

参考运行时间：18.0 s

## 1.3. 运行结果展示

### 1.3.1. o网格生成图片

<center class="half"> 
    <a href=".\ImageExport\ogridrear.png"><img src=".\ImageExport\ogridrear.png" width="280"/></a>
    <a href=".\ImageExport\ogrid.png"><img src=".\ImageExport\ogrid.png" width="280"/></a>
</center>
<p align="center">Fig 1. 翼型后缘网格和全局网格</p>

### 1.3.2. 流场显示



<center class="half">
    <a href=".\ImageExport\cp.png"><img src=".\ImageExport\cp.png" width="280"/></a>
    <a href=".\ImageExport\cpstreamline.png"><img src=".\ImageExport\cpstreamline.png" width="280"/></a>
</center>
<p align="center">Fig 2. 压力系数云图与流线图</p>		

<center class="half">
    <a href=".\ImageExport\u.png"><img src=".\ImageExport\u.png" width="280"/></a>
    <a href=".\ImageExport\v.png"><img src=".\ImageExport\v.png" width="280"/></a>
</center>
<p align="center">Fig 3. x, y方向速度云图</p>		

### 1.3.3. 翼型表面压力系数曲线

![](.\ImageExport\cp-x.png)

<p align="center">Fig 4. 翼型表面压力系数曲线</p>

# 2. 一维Burgers方程数值求解

## 2.1. 输入文档说明

输入文档格式如下：

```csv
time,mu
0.5,0.0
1.2,0.01
2.0,0.05
```

第一列是时间，第二列是粘性系数，可以以$n\times n$的形式输入，即输入$n$个时间，需对应输入$n$个粘性系数。

## 2.2. 运行文档<u>burgers.py</u>说明

该运行文档定义一个类：

```python
class Burgers:
    # 输入粘性系数，最后的计算停止时间，网格空间划分尺度
    def __init__(self, visco, time, N)
    # 对空间和时间进行离散化
    def discretization(self)
    # 使用MacCormack格式进行迭代计算
    def macCormack(self, u)
```

main函数实现：

```python
if __name__ == '__main__':
	N = 200
	seper = os.sep 		# Use this seperator to satisfy both Windows and Linux usage

	# Read input file of time and viscosity
	time_visco = pd.read_csv('input.txt', sep = ',', usecols=['time', 'mu'])
	time = max(time_visco['time'])
	visco = time_visco['mu']

	# Compute Burgers equation at different viscosity
	for i in range(0, len(visco)):
		u = np.zeros(N + 1)
		x = np.zeros(N + 1)

		for j in range(0, N + 1):
			u[j] = -0.5 * (-1 + j * 2 / N)
			x[j] = -1 + j * 2 / N

		burgers = Burgers(visco[i], time, N)
		uall = burgers.macCormack(u)

		# Save 2-D data to files
		output = pd.DataFrame({'x': x, 't='+str(time_visco['time'][0]): uall[int(time_visco['time'][0] * 10000)], \
			't='+str(time_visco['time'][1]): uall[int(time_visco['time'][1] * 10000)], \
			't='+str(time_visco['time'][2]): uall[int(time_visco['time'][2] * 10000)]})
		output.to_csv('2D' + seper + 'mu=' + str(visco[i]) + '.dat', sep='\t', index=False)
```

此外，由于在每次时间步长的计算中记录了计算结果，所以可以将计算结果输出为$x, t, u$格式的三维形式，由于数据量较大，尝试过后发现一个粘性系数的三维数据的输出有160 MB，故直接采用python的matplotlib进行三维图像绘制，仅供参考分析使用。

参考运行时间：

- 2D数据输出：6.0 s
- 2D数据和3D图像输出：20.0 s

## 2.3. 运行结果展示
<center class="half">
    <a href=".\ImageExport\2Dmu=0.0.png"><img src=".\ImageExport\2Dmu=0.0.png" width="280"/></a>
    <a href=".\ImageExport\mu=0.0.png"><img src=".\ImageExport\mu=0.0.png" width="280"/></a>
</center>


<p align="center">Fig 5. $\mu=0.0$ 的数值模拟结果</p>

<center class="half">
    <a href=".\ImageExport\2Dmu=0.01.png"><img src=".\ImageExport\2Dmu=0.01.png" width="280"/></a>
    <a href=".\ImageExport\mu=0.01.png"><img src=".\ImageExport\mu=0.01.png" width="280"/></a>
</center>



<p align="center">Fig 6. $\mu=0.01$ 的数值模拟结果</p>

<center class="half">
    <a href=".\ImageExport\2Dmu=0.05.png"><img src=".\ImageExport\2Dmu=0.05.png" width="280"/></a>
    <a href=".\ImageExport\mu=0.05.png"><img src=".\ImageExport\mu=0.05.png" width="280"/></a>
</center>



<p align="center">Fig 7. $\mu=0.05$ 的数值模拟结果</p>

## 2.4. 结果分析
- 随着时间的推进，速度沿空间的发展越来越陡；
- 无粘情况下 ($\mu=0.0$)速度到最后会发生骤变，梯度有个猛然的变化，并且在速度骤变区域附近会产生振荡；
- $\mu=0.01$时, 速度梯度变化没有无粘情况下那么剧烈，速度变化曲线更为平缓，激波的非物理振荡被消除；
- $\mu=0.05$时，速度梯度变化更小，且在时间推进过程中，速度曲线间的差异越来越小；
- 总结：随着粘性的增大，粘性带来的耗散作用消除了速度的梯度的剧烈变化，并消除了解的振荡；而粘性趋近于零时，解越容易出现激波形式的不连续性。

# 3. 准一维Laval管流动数值求解
## 3.1. 输入文档说明
输入文档格式如下：
```csv
Mach Number,xleft,xright,Pe,Cx,CFL,Errlimit,Iterlimit,xStep
1.16,0.1,1.0,0.8785,0.05,0.5,0.000001,20000,200
```
- Mach Number表示入口马赫数；
- xleft表示入口坐标；
- xright表示出口坐标；
- Pe表示出口压力；
- Cx表示人工粘性系数；
- CFL表示Courant数；
- Errlimit表示最大误差限；
- Iterlimit表示最大迭代次数；
- xStep表示空间离散的空间步。

## 3.2. 运行文档<u>laval.py</u>说明
采用<u>守恒型控制方程</u>进行求解，所以需要使用耦合和解耦

该文档定义一个类：
```python
class Laval():
  # 输入人工粘性，Courant数，空间步长，离散后的空间坐标点，离散后空间截面面积
  def __init__(self, Cx, CFL, dx, x, Ax)
  # 耦合u，p，rho为U，F，H
  def Couples(self, u, p, rho)
  # 将U解耦为u，p，rho
  def deCouples(self, U)
  # 使用MacCormack格式求解
  def macCormack(self, u, p, rho, errlimit, iterlimit)
```
main函数实现：
```python
if __name__ == '__main__':
  sep = os.sep
  input = pd.read_csv('input.txt', sep = ',', usecols=['Mach Number', 'xleft', 'xright', \
  'Pe', 'Cx', 'CFL', 'Errlimit', 'Iterlimit', 'xStep']) # Read csv file as input. Attention! inputs are lists.
  Ma, xl, xr, pe, Cx, CFL, errlimit, iterlimit, N = (input['Mach Number'][0], input['xleft'][0], \
    input['xright'][0], input['Pe'][0], input['Cx'][0], input['CFL'][0], input['Errlimit'][0], \
    input['Iterlimit'][0], input['xStep'][0])

  u, p, rho, x, Ax, dx = initCondition(xl, xr, N, pe, Ma)
  laval = Laval(Cx, CFL, dx, x, Ax)
  u, p, rho = laval.macCormack(u, p, rho, errlimit, iterlimit)
  Ma = u / (np.sqrt(1.4 * p / rho))
  flux = rho * u * Ax

  output = pd.DataFrame({'X/L':x, 'Mach': Ma, 'Velocity':u, 'Pressure':p, \
    'Density':rho, 'Flux':flux})
  output.to_csv('Result' + sep + 'Cx=' + str(Cx) + '.dat', sep='\t', index=False)
```
参考运行时间：5.0 s

## 3.3. 运行结果展示
<center class="half">
    <a href=".\ImageExport\cx0.05Ma.png"><img src=".\ImageExport\cx0.05Ma.png" width="280"/></a>
    <a href=".\ImageExport\cx0.05u.png"><img src=".\ImageExport\cx0.05u.png" width="280"/></a>
</center>



<p align="center">Fig 8. $C_x=0.05$ 的数值模拟结果</p>

<center class="half">
    <a href=".\ImageExport\cx0.15Ma.png"><img src=".\ImageExport\cx0.15Ma.png" width="280"/></a>
    <a href=".\ImageExport\cx0.15u.png"><img src=".\ImageExport\cx0.15u.png" width="280"/></a>
</center>



<p align="center">Fig 9. $C_x=0.15$ 的数值模拟结果</p>

<center class="half">
    <a href=".\ImageExport\cx0.2Ma.png"><img src=".\ImageExport\cx0.2Ma.png" width="280"/></a>
    <a href=".\ImageExport\cx0.2u.png"><img src=".\ImageExport\cx0.2u.png" width="280"/></a>
</center>



<p align="center">Fig 10. $C_x=0.20$ 的数值模拟结果</p>
## 3.4. 结果分析
### 3.4.1. 人工粘性系数的影响
从上面的图中可以看到：

- 人工粘性系数为0.05的时候，激波附近出现高频非物理振荡，沿管内流量不变；
- 人工粘性系数为0.15时，激波附近的高频非物理振荡得到抑制，沿管道内流量不变；
- 人工粘性系数为0.20时，激波附近的高频非物理振荡几乎得到抑制，沿管道内流量不变；
- 总结：随着人工粘性系数的增大，激波间断点处的梯度有所减小，斜率略有改变和倾斜，解的精度有所降低。
### 3.4.2. 与解析解比较
<center class="half">
    <a href=".\ImageExport\cx0.05compare.png"><img src=".\ImageExport\cx0.05compare.png" width="280"/></a>
    <a href=".\ImageExport\cx0.15compare.png"><img src=".\ImageExport\cx0.15compare.png" width="280"/></a>
    <a href=".\ImageExport\cx0.2compare.png"><img src=".\ImageExport\cx0.2compare.png" width="280"/></a>
</center>



<p align="center">Fig 11. 数值模拟结果和解析解的比较</p>
可以看出，随着人工粘性的增大，数值解有所发散，和解析解的差别有所增大，表明解的精度确实有所降低，人工粘性确实有数值耗散的作用。