import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# ------------------------------------------------------------
# 1. Загрузка данных с правильной кодировкой
# ------------------------------------------------------------
file_path = 'data/baskets.csv'
if not os.path.exists(file_path):
    file_path = 'baskets.csv'

# Читаем с кодировкой cp1251 (русская Windows)
try:
    data = pd.read_csv(file_path, header=None, names=['items'], encoding='cp1251')
except UnicodeDecodeError:
    # если не получается, пробуем другие кодировки
    data = pd.read_csv(file_path, header=None, names=['items'], encoding='utf-8-sig')

data = data.dropna()
transactions = data['items'].apply(
    lambda x: [item.strip() for item in x.split(',') if item.strip() != '']
).tolist()

print(f"Всего транзакций: {len(transactions)}")
print("Пример первых 5 транзакций:")
for i in range(min(5, len(transactions))):
    print(transactions[i])

# ------------------------------------------------------------
# 2. One-hot encoding
# ------------------------------------------------------------
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# ------------------------------------------------------------
# 3. Эксперименты
# ------------------------------------------------------------
support_thresholds = [0.01, 0.03, 0.05, 0.10, 0.15]
results = {}

for min_sup in support_thresholds:
    print(f"\n--- Запуск для порога {min_sup*100:.0f}% ---")
    start = time.perf_counter()
    frequent_itemsets = apriori(df, min_support=min_sup, use_colnames=True)
    elapsed = time.perf_counter() - start
    print(f"Время выполнения: {elapsed:.4f} сек")
    print(f"Найдено частых наборов: {len(frequent_itemsets)}")

    if not frequent_itemsets.empty:
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
        length_counts = frequent_itemsets['length'].value_counts().sort_index().to_dict()
    else:
        length_counts = {}

    results[min_sup] = {
        'time': elapsed,
        'itemsets': frequent_itemsets,
        'length_counts': length_counts
    }

# ------------------------------------------------------------
# 4. Сохранение результатов
# ------------------------------------------------------------
os.makedirs('results/frequent_itemsets', exist_ok=True)
for min_sup, res in results.items():
    if not res['itemsets'].empty:
        df_out = res['itemsets'][['itemsets', 'support']].copy()
        df_out['itemsets'] = df_out['itemsets'].apply(lambda x: list(x))
        filename = f'results/frequent_itemsets/support_{int(min_sup*100)}.csv'
        df_out.to_csv(filename, index=False, encoding='cp1251')
        print(f"Сохранено в {filename}")
    else:
        print(f"Для порога {min_sup*100:.0f}% частых наборов нет.")

# ------------------------------------------------------------
# 5. График времени
# ------------------------------------------------------------
os.makedirs('results/charts', exist_ok=True)

thresholds = list(results.keys())
times = [results[t]['time'] for t in thresholds]

plt.figure(figsize=(10, 6))
plt.plot(thresholds, times, marker='o', linestyle='-', linewidth=2, color='blue')
plt.xlabel('Порог поддержки (min_support)')
plt.ylabel('Время выполнения (сек)')
plt.title('Зависимость времени выполнения алгоритма Apriori от порога поддержки')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('results/charts/time_vs_support.png', dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------
# 6. График количества наборов по длине
# ------------------------------------------------------------
max_len = 0
for res in results.values():
    if res['length_counts']:
        max_len = max(max_len, max(res['length_counts'].keys()))

lengths = range(1, max_len + 1)

fig, ax = plt.subplots(figsize=(12, 6))
width = 0.15
x = np.arange(len(lengths))

for i, (threshold, res) in enumerate(results.items()):
    counts = [res['length_counts'].get(l, 0) for l in lengths]
    ax.bar(x + i * width, counts, width, label=f'{threshold*100:.0f}%')

ax.set_xlabel('Длина набора (количество товаров)')
ax.set_ylabel('Количество частых наборов')
ax.set_title('Распределение частых наборов по длине при разных порогах поддержки')
ax.set_xticks(x + width * (len(results) - 1) / 2)
ax.set_xticklabels(lengths)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('results/charts/itemsets_by_length.png', dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------
# 7. Пример сортировки (для порога 5%)
# ------------------------------------------------------------
min_sup_example = 0.05
itemsets_example = results[min_sup_example]['itemsets']
if not itemsets_example.empty:
    sorted_by_support = itemsets_example.sort_values('support', ascending=False)
    print(f"\nТоп-10 наборов по поддержке для порога {min_sup_example*100:.0f}%:")
    print(sorted_by_support[['itemsets', 'support']].head(10))

    itemsets_example['itemsets_tuple'] = itemsets_example['itemsets'].apply(lambda x: tuple(sorted(x)))
    sorted_lex = itemsets_example.sort_values('itemsets_tuple')
    print("\nПервые 10 наборов в лексикографическом порядке:")
    print(sorted_lex[['itemsets', 'support']].head(10))
else:
    print("Для порога 5% нет частых наборов.")