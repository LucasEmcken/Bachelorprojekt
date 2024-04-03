import numpy as np

def pascal(x):
    triangle = []
    for i in range(x):
        row = [1]
        if i > 0:
            prev_row = triangle[-1]
            for j in range(len(prev_row) - 1):
                row.append(prev_row[j] + prev_row[j + 1])
            row.append(1)
        triangle.append(row)
    return triangle[-1]

def lorentzian(x, mean, variance):
    return 1 / (np.pi * variance * (1 + ((x - mean) / variance) ** 2))
def gauss(x, mean, variance):
    return 1/(variance*np.sqrt(2*np.pi))*np.exp(-1/2*((x-mean)/variance)**2)

def multiplet(x, mult, mean, sigma, spacing, type='lorentz'):
    triangle = pascal(mult)
    t_max = max(triangle)
    triangle = [t/t_max for t in triangle]
    y = np.zeros(len(x),dtype=float)

    if len(triangle)%2 == 0:
        space = -1*len(triangle)/2*spacing+spacing/2
    else:
        space = -1*(len(triangle)-1)/2*spacing
    for i,size in enumerate(triangle):
        if type == 'lorentz':
            y += lorentzian(x, mean+space, sigma)*size
        else:
            y += gauss(x, mean+space, sigma)*size
        space +=  spacing
    return y

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = np.linspace(-100,100,1000)
    Y = multiplet(x,5, 5, 1, 20)

    plt.plot(x, Y)
    plt.title('Lorentzian Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()