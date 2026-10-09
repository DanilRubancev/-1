# main.py
"""
Основной скрипт: запуск экспериментов на baskets.csv,
построение диаграмм, сохранение результатов.
"""

import os
from collections import Counter
import matplotlib
matplotlib.use('TkAgg')   # чтобы графики точно открывались в отдельном окне

import matplotlib.pyplot as plt

from apriori import load_transactions, run_apriori_with_time


# ------------------ Параметры экспериментов ------------------
# Если файл называется по-другому, поправьте путь:
CSV_PATH = 'baskets (1).csv'

# Можно указать полный путь (раскомментируйте, если нужно):
# CSV_PATH = r'C:\Users\Петр\PycharmProjects\yaziki progr\baskets (1).csv'

THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.15]   # 1%, 3%, 5%, 10%, 15%
SORT_BY = 'support'    # 'support' или 'lex'
# -------------------------------------------------------------


def main():
    # 1. Загрузка данных
    if not os.path.exists(CSV_PATH):
        print(f"Файл '{CSV_PATH}' не найден.")
        print("Положите baskets (1).csv рядом с main.py или укажите полный путь в CSV_PATH.")
        return

    transactions = load_transactions(CSV_PATH)
    print(f"Загружено транзакций: {len(transactions)}")
    unique_items = set()
    for t in transactions:
        unique_items.update(t)
    print(f"Уникальных товаров: {len(unique_items)}")
    print("-" * 60)

    # 2. Проведение экспериментов
    times = []
    counts_by_length = {t: Counter() for t in THRESHOLDS}
    total_counts = []

    for thr in THRESHOLDS:
        print(f"Порог поддержки: {thr*100:.0f}% ...", end=' ', flush=True)
        result, elapsed = run_apriori_with_time(transactions, thr, sort_by=SORT_BY)
        times.append(elapsed)
        total_counts.append(len(result))

        for itemset, support, count in result:
            counts_by_length[thr][len(itemset)] += 1

        print(f"время = {elapsed:.3f} с, найдено наборов: {len(result)}")

        # Сохраняем результаты для каждого порога в отдельный файл
        with open(f'result_{int(thr*100)}.txt', 'w', encoding='utf-8') as f:
            f.write(f"# Порог поддержки: {thr*100:.0f}%\n")
            f.write(f"# Всего транзакций: {len(transactions)}\n")
            f.write(f"# Найдено наборов: {len(result)}\n")
            f.write("# набор | поддержка (доля) | поддержка (кол-во)\n")
            for itemset, support, count in result:
                f.write(f"{', '.join(sorted(itemset))} | {support:.4f} | {count}\n")

    print("-" * 60)
    print("Эксперименты завершены. Результаты сохранены в result_*.txt")

    # 3. Диаграмма 1: время работы vs порог поддержки
    plt.figure(figsize=(8, 5))
    x_labels = [f"{int(t*100)}%" for t in THRESHOLDS]
    plt.plot(x_labels, times, marker='o', color='tab:blue', linewidth=2)
    plt.title('Зависимость времени работы Apriori от порога поддержки')
    plt.xlabel('Порог поддержки')
    plt.ylabel('Время работы (сек)')
    plt.grid(True, linestyle='--', alpha=0.6)
    for i, t in enumerate(times):
        plt.annotate(f"{t:.2f}", (x_labels[i], times[i]),
                     textcoords="offset points", xytext=(0, 8), ha='center')
    plt.tight_layout()
    plt.savefig('chart_time.png', dpi=150)
    plt.show()

    # 4. Диаграмма 2: количество частых наборов по длине
    max_len = max((max(c.keys()) if c else 0) for c in counts_by_length.values())
    max_len = max(max_len, 1)

    plt.figure(figsize=(10, 6))
    bar_width = 0.15
    x_positions = list(range(1, max_len + 1))

    for idx, thr in enumerate(THRESHOLDS):
        heights = [counts_by_length[thr].get(length, 0) for length in x_positions]
        offsets = [x + (idx - len(THRESHOLDS) / 2) * bar_width + bar_width / 2
                   for x in x_positions]
        plt.bar(offsets, heights, width=bar_width, label=f"{int(thr*100)}%")

    plt.title('Количество частых наборов различной длины при разных порогах поддержки')
    plt.xlabel('Длина набора (число товаров)')
    plt.ylabel('Количество наборов')
    plt.xticks(x_positions)
    plt.legend(title='Порог поддержки')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('chart_lengths.png', dpi=150)
    plt.show()

    # 5. Диаграмма 3 (дополнительно): общее количество наборов vs порог
    plt.figure(figsize=(8, 5))
    plt.bar(x_labels, total_counts, color='tab:green')
    plt.title('Общее количество частых наборов при разных порогах поддержки')
    plt.xlabel('Порог поддержки')
    plt.ylabel('Количество наборов')
    for i, c in enumerate(total_counts):
        plt.annotate(str(c), (i, c), textcoords="offset points",
                     xytext=(0, 4), ha='center')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('chart_total.png', dpi=150)
    plt.show()

    # 6. Топ-10 наборов при пороге 3% (для наглядности в отчёте)
    print("\nТоп-10 частых наборов при пороге 3%:")
    result_3, _ = run_apriori_with_time(transactions, 0.03, sort_by='support')
    for itemset, support, count in result_3[:10]:
        print(f"  {', '.join(sorted(itemset)):60s} support={support:.4f} ({count})")


if __name__ == '__main__':
    main()