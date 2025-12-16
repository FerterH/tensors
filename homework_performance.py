import torch
import time
import pandas as pd

### 3.1 Подготовка данных (5 баллов)
# Создаем тензоры на CPU
t1 = torch.randn(64, 1024, 1024, dtype=torch.float32)
t2 = torch.randn(128, 512, 512, dtype=torch.float32)
t3 = torch.randn(256, 256, 256, dtype=torch.float32)


### 3.2 Функция измерения времени (5 баллов)
def measure_time_cpu(func, *args, num_runs=10, **kwargs):


    # Измерение среднего времени
    start_time = time.perf_counter()
    for _ in range(num_runs):
        result = func(*args, **kwargs)
    end_time = time.perf_counter()

    elapsed_ms = ((end_time - start_time) / num_runs) * 1000.0
    return elapsed_ms, result


def measure_time_gpu(func, *args, num_runs=10, **kwargs):
    torch.cuda.synchronize()

    # Измерение среднего времени
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    for _ in range(num_runs):
        result = func(*args, **kwargs)

    end_event.record()
    end_event.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / num_runs
    return elapsed_ms, result


### 3.3 Сравнение операций (10 баллов)
# Проверяем доступность GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Создаем копии тензоров для GPU
if device == 'cuda':
    t1_gpu = t1.to(device)
    t2_gpu = t2.to(device)
    t3_gpu = t3.to(device)

# Инициализация списка для хранения результатов
results = []

# Операции на CPU
cpu_operations = [
    ("Матричное умножение", torch.matmul, (t1, t1.transpose(-1, -2))),
    ("Поэлементное сложение", torch.add, (t1, t1)),
    ("Поэлементное умножение", torch.mul, (t1, t1)),
    ("Транспонирование", torch.transpose, (t1, 0, 1)),
    ("Сумма всех элементов", torch.sum, (t1,))
]

# Операции на GPU (если доступно)
if device == 'cuda':
    gpu_operations = [
        ("Матричное умножение", torch.matmul, (t1_gpu, t1_gpu.transpose(-1, -2))),
        ("Поэлементное сложение", torch.add, (t1_gpu, t1_gpu)),
        ("Поэлементное умножение", torch.mul, (t1_gpu, t1_gpu)),
        ("Транспонирование", torch.transpose, (t1_gpu, 0, 1)),
        ("Сумма всех элементов", torch.sum, (t1_gpu,))
    ]
else:
    gpu_operations = []

# Измеряем время на CPU
cpu_results = {}
for op_name, op_func, op_args in cpu_operations:
    cpu_ms, _ = measure_time_cpu(op_func, *op_args)
    cpu_results[op_name] = cpu_ms
    print(f"{op_name}: {cpu_ms:.3f} мс")

# Измеряем время на GPU (если доступно)
gpu_results = {}
if device == 'cuda':
    print("\nИзмерения на GPU:")
    for op_name, op_func, op_args in gpu_operations:
        gpu_ms, _ = measure_time_gpu(op_func, *op_args)
        gpu_results[op_name] = gpu_ms
        print(f"{op_name}: {gpu_ms:.3f} мс")

# Собираем результаты в таблицу
for op_name in [op[0] for op in cpu_operations]:
    cpu_ms = cpu_results[op_name]

    if op_name in gpu_results:
        gpu_ms = gpu_results[op_name]
        # Ускорение = CPU_time / GPU_time (больше 1 означает ускорение)
        if gpu_ms > 0:
            speedup = cpu_ms / gpu_ms
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
    else:
        gpu_ms = "N/A"
        speedup_str = "N/A"

    results.append({
        "Операция": op_name,
        "CPU (мс)": f"{cpu_ms:.1f}",
        "GPU (мс)": f"{gpu_ms:.1f}" if isinstance(gpu_ms, float) else gpu_ms,
        "Ускорение": speedup_str
    })
