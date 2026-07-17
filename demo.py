import matplotlib.pyplot as plt

x=[1,2,3,4,5]
y=[10,20,15,30,25]

plt.plot(x, y, marker='o')

plt.title("Line Graph")
plt.xlabel("x values")
plt.ylabel("y values")
plt.grid(True)

plt.show()