from matplotlib import pyplot as plt
import matplotlib

def amdahls_law(p, s):
    """Speedup relative to proportion parallel
    Amdahl's Law gives an idealized speedup we
    can expect for an algorithm given the proportion
    that algorithm can be parallelized and the speed
    we gain from that parallelization. The best case
    scenario is that the speedup, `s`, is equal to
    the number of processors available.
    Args:
        p: proportion parallel
        s: speed up for the parallelized proportion
    """
    return 1 / ((1-p) + p/s)


def distances_plot(p):
    x = []
    y = []
    for s in range(1, 65537, 10):
        x.append(s)
        y.append(amdahls_law(p, s))
    print(y[-1])
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, label=f'Theoretical Speedup for P={p*100}%', color='mediumseagreen')
    ax.axhline(y=130.129, label=f'Jacob\'s Vectorized', color='dodgerblue')
    ax.axhline(y=223.950, label=f'My Vectorized', color='orangered')
    ax.set_title('Amdahl\'s Law for Centroid Distances')
    ax.set_xlabel('Number of Processors (P) in log scale')
    ax.set_ylabel('SpeedUp')
    ax.legend(loc='upper left')
    ax.set_xscale('log')
    ax.set_xticks([16, 64, 256, 1024, 4096, 16384, 65536])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.show()
    fig.savefig('../outputs/final/assignment3/results/question2.png')
    plt.clf()


distances_plot(p=0.9982)