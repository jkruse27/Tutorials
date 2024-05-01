from utils.utils import read_file
from utils.dma import create_scales, dma
import matplotlib.pyplot as plt

hrv = read_file('data/example_hrv.csv')

scales = create_scales(5, 90000)
out = dma(hrv, scales, 2)


plt.plot(scales, out)
plt.xscale('log')
plt.yscale('log')
plt.show()