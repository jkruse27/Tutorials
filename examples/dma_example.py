import matplotlib.pyplot as plt
import time
from hrv_utils.utils import read_file
from hrv_utils.dma import create_scales, dma

# Read example time series
start = time.time()
hrv = read_file('data/example_hrv.csv')
end = time.time()
print(f'Data size: {len(hrv)} points')
print(f'Data reading time: {end-start: .5f}s')

# Generate scales that will be used
start = time.time()
scales_0 = create_scales(3, len(hrv)//4)
# Compute DMA
out_0 = dma(hrv, scales_0, 0)
end = time.time()
print(f'0 order DMA processing time: {end-start: .5f}s')

# Generate scales that will be used
start = time.time()
scales_2 = create_scales(5, len(hrv)//2)
# Compute DMA
out_2 = dma(hrv, scales_2, 2)
end = time.time()
print(f'2nd order DMA processing time: {end-start: .5f}s')

# Generate scales that will be used
start = time.time()
scales_4 = create_scales(5, len(hrv)//2)
# Compute DMA
out_4 = dma(hrv, scales_4, 4)
end = time.time()
print(f'4th order DMA processing time: {end-start: .5f}s')

# Plot results
plt.scatter(scales_0, out_0, label='order = 0')
plt.scatter(scales_2, out_2, label='order = 2')
plt.scatter(scales_4, out_4, label='order = 4')
plt.xlabel('$log_{10}$ s')
plt.ylabel('$log_{10}$ F(s)')

plt.title('DMA for the time series')
plt.xscale('log')
plt.yscale('log')
plt.gca().set_aspect(1.25)
plt.legend()
plt.show()
