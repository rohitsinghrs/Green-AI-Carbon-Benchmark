from codecarbon import EmissionsTracker
import time

print("=" * 60)
print("CODECARBON TEST")
print("=" * 60)

tracker = EmissionsTracker(
    project_name="green_ai_hardware_test",
    output_dir="codecarbon_test"
)

tracker.start()

start = time.perf_counter()

# Small controlled workload
total = 0

for i in range(5_000_000):
    total += i * i

elapsed = time.perf_counter() - start

emissions = tracker.stop()

print("\nWorkload completed")

print("Execution time:", round(elapsed, 4), "seconds")

print(
    "Emissions:",
    emissions,
    "kg CO2eq"
)

print("\nCheck folder:")
print("codecarbon_test/")

print("=" * 60)