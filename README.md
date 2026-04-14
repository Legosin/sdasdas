python
"""
Обработка экспериментальных данных методом наименьших квадратов.

Скрипт загружает данные из CSV-файла, выполняет линейную и полиномиальную
аппроксимацию, вычисляет метрики качества и строит графики.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
import sys
from pathlib import Path


def load_data(file_path):
    """Загрузка данных из CSV. Ожидаются столбцы 'x' и 'y'."""
    x, y = [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x.append(float(row['x']))
                y.append(float(row['y']))
    except Exception as e:
        print(f"Ошибка чтения файла {file_path}: {e}")
        sys.exit(1)
    return np.array(x), np.array(y)


def linear_func(x, a, b):
    """Линейная функция y = a*x + b."""
    return a * x + b


def polynomial_func(x, *coeffs):
    """Полином произвольной степени."""
    return np.polyval(coeffs, x)


def r_squared(y_true, y_pred):
    """Коэффициент детерминации R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def rmse(y_true, y_pred):
    """Среднеквадратичная ошибка."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def fit_and_plot(x, y, degree=1, output_plot='fit_plot.png'):
    """
    Выполняет аппроксимацию полиномом заданной степени,
    выводит параметры, метрики и сохраняет график.
    """
    # Аппроксимация полиномом (для degree=1 эквивалентно линейной регрессии)
    coeffs, cov = np.polyfit(x, y, degree, cov=True)
    y_fit = np.polyval(coeffs, x)

    # Дополнительно используем curve_fit для получения ошибок параметров
    if degree == 1:
        popt, pcov = curve_fit(linear_func, x, y)
        a, b = popt
        a_err, b_err = np.sqrt(np.diag(pcov))
        equation = f'y = ({a:.4f} ± {a_err:.4f})·x + ({b:.4f} ± {b_err:.4f})'
    else:
        popt, pcov = curve_fit(lambda x, *params: polynomial_func(x, *params),
                               x, y, p0=coeffs)
        errors = np.sqrt(np.diag(pcov))
        equation = 'y = ' + ' + '.join([f'({c:.4f} ± {e:.4f})·x^{degree-i}'
                                        for i, (c, e) in enumerate(zip(popt, errors))])
        a, b = None, None

    # Метрики
    r2 = r_squared(y, y_fit)
    err = rmse(y, y_fit)

    print(f"\nРезультаты аппроксимации полиномом степени {degree}:")
    print(f"Уравнение: {equation}")
    print(f"R² = {r2:.6f}")
    print(f"RMSE = {err:.6f}")

    # Построение графика
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Экспериментальные данные', alpha=0.7)
    x_smooth = np.linspace(min(x), max(x), 200)
    y_smooth = np.polyval(coeffs, x_smooth)
    plt.plot(x_smooth, y_smooth, 'r-', label=f'Полином степени {degree}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Метод наименьших квадратов (степень {degree})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"График сохранён в '{output_plot}'")

    return coeffs, r2, err


def main():
    # Путь к файлу данных (можно передать аргументом командной строки)
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = 'data.csv'

    if not Path(data_file).exists():
        print(f"Файл '{data_file}' не найден. Создаю пример данных...")
        # Генерация тестовых данных с шумом
        np.random.seed(42)
        x = np.linspace(0, 10, 20)
        y = 2.5 * x + 1.0 + np.random.normal(0, 1.5, size=len(x))
        # Сохранение примера
        with open('data.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            for xi, yi in zip(x, y):
                writer.writerow([xi, yi])
        print("Создан файл 'data.csv' с примером данных.")
        data_file = 'data.csv'

    # Загрузка данных
    x, y = load_data(data_file)
    print(f"Загружено {len(x)} точек из '{data_file}'")

    # Линейная регрессия (степень 1)
    fit_and_plot(x, y, degree=1, output_plot='linear_fit.png')

    # Полиномиальная регрессия степени 2 (можно изменить)
    fit_and_plot(x, y, degree=2, output_plot='quadratic_fit.png')


if __name__ == '__main__':
    main()# sdasdas
