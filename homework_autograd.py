import torch
### 2.1 Простые вычисления с градиентами (8 баллов)
# Создаем тензоры с requires_grad=True
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)

# Вычисляем функцию f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = x**2 + y**2 + z**2 + 2*x*y*z

# Вычисляем градиенты
f.backward()

# Выводим результаты
print(f"Значение функции: {f.item()}")
print(f"df/dx: {x.grad}")
print(f"df/dy: {y.grad}")
print(f"df/dz: {z.grad}")
# Было подсчитано на калькуляторе - ответы сходятся


### 2.2 Градиент функции потерь (9 баллов)
# Пример данных (векторные входы и целевые значения)
x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])

# Параметры модели (требуют градиента)
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Линейная модель и MSE
y_pred = w * x + b
mse = ((y_pred - y_true) ** 2).mean()

# Вычисление градиентов
mse.backward()

# Результаты
print(f"MSE: {mse.item()}")
print(f"df/dw: {w.grad}")
print(f"df/db: {b.grad}")

### 2.3 Цепное правило (8 баллов)
# f(x) = sin(x^2 + 1)
# создаём скалярный тензор x с необходимостью вычисления градиента
x = torch.tensor(1.2, requires_grad=True)

# вычисляем функцию
f = torch.sin(x**2 + 1)

# градиент через torch.autograd.grad (возвращает кортеж)
grad_autograd = torch.autograd.grad(f, x)[0]

# градиент через backward (запишется в x.grad)
f.backward()

# аналитический градиент: df/dx = 2*x * cos(x^2 + 1)
grad_analytic = 2 * x * torch.cos(x**2 + 1)

# выводим результаты
print(f"f: {f.item()}")
print(f"df/dx (x.grad): {x.grad.item()}")
print(f"df/dx (autograd.grad): {grad_autograd.item()}")
print(f"df/dx (analytic): {grad_analytic.item()}")

