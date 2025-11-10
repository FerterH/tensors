import torch


# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
random_tensor = torch.rand(3, 4)
print(random_tensor)
# - Тензор размером 2x3x4, заполненный нулями
zero_tenzor = torch.zeros(2, 3, 4)
print(zero_tenzor)
# - Тензор размером 5x5, заполненный единицами
ones_tenzor = torch.ones(5, 5)
# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
range_tensor = torch.arange(16).reshape(4, 4)
print(range_tensor)

# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
range_tensor = torch.arange(16).reshape(4, 4)
print(range_tensor)

### 1.2 Операции с тензорами (6 баллов)
# Создаем тензоры A и B
A = torch.rand(3, 4)
B = torch.rand(4, 3)

# Транспонирование тензора A
A_transposed = A.transpose(0, 1)
print("Транспонированный A:")
print(A_transposed)

# Матричное умножение A и B
matrix_mult = torch.matmul(A, B)
print("Матричное умножение A и B:")
print(matrix_mult)

# Поэлементное умножение A и транспонированного B
element_wise_mult = A * B.transpose(0, 1)
print("Поэлементное умножение A и транспонированного B:")
print(element_wise_mult)

# Сумма всех элементов тензора A
sum_A = torch.sum(A)
print("Сумма всех элементов тензора A:")
print(sum_A)

### 1.3 Индексация и срезы (6 баллов)
# Создайте тензор размером 5x5x5
# Буду использовать значения от 0 до 124, чтобы было проще визуально проверять извлечения
t = torch.arange(125).reshape(5, 5, 5)
print("Тензор t (5x5x5):")
print(t)

# - Первую строку (берём первую строку первой 2D-матрицы t[0])
first_row = t[0, 0, :]
print("Первая строка (t[0,0,:]):")
print(first_row)

# - Последний столбец (последний столбец первой 2D-матрицы t[0])
last_col = t[0, :, -1]
print("Последний столбец (t[0,:, -1]):")
print(last_col)

# - Подматрица размером 2x2 из центра тензора (внутри первой 2D-матрицы)
# Для размера 5x5 центральные индексы близкие к середине: возьмём индексы 1:3
center_2x2 = t[0, 1:3, 1:3]
print("Центральная подматрица 2x2 (t[0,1:3,1:3]):")
print(center_2x2)

# - Все элементы с четными индексами (шаг 2 по каждой оси)
even_indices = t[::2, ::2, ::2]
print("Элементы с четными индексами (t[::2,::2,::2]):")
print(even_indices)

### 1.4 Работа с формами (6 баллов)
# Создайте тензор размером 24 элемента
v = torch.arange(24)
print("Вектор v (24 элементов):")
print(v)

# Преобразуйте его в формы:
reshaped_2x12 = v.reshape(2, 12)
print("Форма 2x12:", reshaped_2x12.shape)
print(reshaped_2x12)

reshaped_3x8 = v.reshape(3, 8)
print("Форма 3x8:", reshaped_3x8.shape)
print(reshaped_3x8)

reshaped_4x6 = v.reshape(4, 6)
print("Форма 4x6:", reshaped_4x6.shape)
print(reshaped_4x6)

reshaped_2x3x4 = v.reshape(2, 3, 4)
print("Форма 2x3x4:", reshaped_2x3x4.shape)
print(reshaped_2x3x4)

reshaped_2x2x2x3 = v.reshape(2, 2, 2, 3)
print("Форма 2x2x2x3:", reshaped_2x2x2x3.shape)
print(reshaped_2x2x2x3)
