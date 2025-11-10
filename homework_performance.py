import torch
import time
import pandas as pd

### 3.1 Подготовка данных (5 баллов)
# 64 x 1024 x 1024
t1 = torch.randn(64, 1024, 1024, dtype=torch.float32)

# 128 x 512 x 512
t2 = torch.randn(128, 512, 512, dtype=torch.float32)

# 256 x 256 x 256
t3 = torch.randn(256, 256, 256, dtype=torch.float32)


### 3.2 Функция измерения времени (5 баллов)
def measure_time_cpu(func, *args, **kwargs):
    start_time = time.time()
    _ = func(*args, **kwargs)
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000.0
    print(f"CPU : {elapsed_ms:.3f} мс")
    return elapsed_ms


def measure_time_gpu(func, *args, **kwargs):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    _ = func(*args, **kwargs)
    end_event.record()

    end_event.synchronize()
    elapsed = start_event.elapsed_time(end_event)
    print(f"GPU : {elapsed:.3f} мс")
    return elapsed


### 3.3 Сравнение операций (10 баллов)
# Перемещение тензоров на GPU, если доступен
device = 'cuda' if torch.cuda.is_available() else 'cpu'
t1_gpu = t1.to(device)
t2_gpu = t2.to(device)
t3_gpu = t3.to(device)

# Инициализация списка для хранения результатов
results = []

# Определение операций для сравнения (используем совместимые тензоры)
operations = [
    ("Матричное умножение", torch.matmul, (t1_gpu, t1_gpu.transpose(-1, -2))),
    ("Поэлементное сложение", torch.add, (t1_gpu, t1_gpu)),
    ("Поэлементное уумножение", torch.mul, (t1_gpu, t1_gpu)),
    ("Транспонирование", torch.transpose, (t1_gpu, 0, 1)),
    ("Сумма всех элементов", torch.sum, (t1_gpu,))
]

# Измерение времени для каждой операции (только один запуск измерений)
for op_name, op_func, op_args in operations:
    cpu_ms = measure_time_cpu(op_func, *op_args)
    if device == 'cuda':
        gpu_ms = measure_time_gpu(op_func, *op_args)
        speedup = cpu_ms / gpu_ms if gpu_ms and gpu_ms > 0 else None
    else:
        gpu_ms = None
        speedup = None

    results.append({
        "Операция": op_name,
        "CPU мс": cpu_ms,
        "GPU мс": gpu_ms if gpu_ms is not None else "N/A",
        "Ускорение": f"{speedup:.2f}x" if speedup is not None else "N/A"
    })

# Показать результаты
results_df = pd.DataFrame(results)
print(results_df)
