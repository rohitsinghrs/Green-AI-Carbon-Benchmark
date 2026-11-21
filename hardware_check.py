import platform
import psutil

print("=" * 50)
print("HARDWARE INFORMATION")
print("=" * 50)

print("OS:", platform.platform())
print("Machine:", platform.machine())
print("Processor:", platform.processor())

print("\nCPU")
print("Physical cores:", psutil.cpu_count(logical=False))
print("Logical cores:", psutil.cpu_count(logical=True))

print("\nRAM")
ram_gb = psutil.virtual_memory().total / (1024 ** 3)
print("Total RAM: %.2f GB" % ram_gb)

print("\nCPU frequency:")
print(psutil.cpu_freq())

print("=" * 50)