from matplotlib import pyplot as plt

fig: plt.Figure = plt.figure()
plt.style.use('fivethirtyeight')
ax: plt.Axes = fig.add_subplot()
x = [0, 10, 20, 30, 40, 50]
y = [0, 70, 60, 90, 88, 30]
ax.plot(x, y, linestyle='-', color='blue')
ax.set_title("test57 line")
ax.set_xlabel("X label")
ax.set_ylabel("Y label")
ax.set_facecolor("green")
plt.show()
