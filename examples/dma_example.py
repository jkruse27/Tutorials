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
scales = create_scales(5, len(hrv)//2)
end = time.time()
print(f'Scales creation time: {end-start: .5f}s')

# Compute DMA
start = time.time()
out = dma(hrv, scales, 0)
end = time.time()
print(f'0th order DMA processing time: {end-start: .5f}s')

# Plot results
plt.scatter(scales, out)
plt.xlabel('$log_{10}$ s')
plt.ylabel('$log_{10}$ F(s)')
plt.title('DMA for the time series')
plt.xscale('log')
plt.yscale('log')
plt.show()
