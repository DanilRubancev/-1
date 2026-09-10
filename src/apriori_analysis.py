import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# ------------------------------------------------------------
# 1. Загрузка данных
# ------------------------------------------------------------
file_path = 'data/baskets.csv'
if not os.path.exists(file_path):
    file_path = 'baskets.csv'

for enc in ('utf-8-sig', 'utf-8', 'cp1251'):
    try:
        with open(file_path, 'r', encoding=enc) as f:
            lines = f.readlines()
        break
    except UnicodeDecodeError:
        continue

transactions = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    items = [item.strip() for item in line.split(',') if item.strip()]
    if items:
        transactions.append(items)

print(f"Транзакций: {len(transactions)}")

# ------------------------------------------------------------
# 2. One-hot encoding
# ------------------------------------------------------------
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(f"Уникальных товаров: {len(te.columns_)}")

# ------------------------------------------------------------
# 3. Эксперименты
# ------------------------------------------------------------
support_thresholds = [0.01, 0.03, 0.05, 0.10, 0.15]
results = {}

for min_sup in support_thresholds:
    start = time.perf_counter()
    frequent_itemsets = apriori(df, min_support=min_sup, use_colnames=True)
    elapsed = time.perf_counter() - start

    if not frequent_itemsets.empty:
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
        length_counts = frequent_itemsets['length'].value_counts().sort_index().to_dict()
    else:
        length_counts = {}

    results[min_sup] = {
        'time': elapsed,
        'itemsets': frequent_itemsets,
        'length_counts': length_counts,
    }
    print(f"Порог {min_sup*100:.0f}%: время {elapsed:.4f} сек, "
          f"наборов {len(frequent_itemsets)}, по длине {length_counts}")

# ------------------------------------------------------------
# 4. Сохранение результатов
# ------------------------------------------------------------
os.makedirs('results/frequent_itemsets', exist_ok=True)
for min_sup, res in results.items():
    if not res['itemsets'].empty:
        df_out = res['itemsets'][['itemsets', 'support']].copy()
        df_out['itemsets'] = df_out['itemsets'].apply(lambda x: list(x))
        filename = f'results/frequent_itemsets/support_{int(min_sup*100)}.csv'
        df_out.to_csv(filename, index=False, encoding='utf-8-sig')

# ------------------------------------------------------------
# 5. График времени выполнения
# ------------------------------------------------------------
os.makedirs('results/charts', exist_ok=True)
thresholds = list(results.keys())
times = [results[t]['time'] for t in thresholds]

plt.figure(figsize=(10, 6))
plt.plot(thresholds, times, marker='o', linestyle='-', linewidth=2, color='blue')
for x, y in zip(thresholds, times):
    plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=9)
plt.xlabel('Порог поддержки (min_support)')
plt.ylabel('Время выполнения (сек)')
plt.title('Зависимость времени выполнения алгоритма Apriori от порога поддержки')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('results/charts/time_vs_support.png', dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------
# 6. График количества частых наборов по длине
# ------------------------------------------------------------
all_lengths = set()
for res in results.values():
    all_lengths.update(res['length_counts'].keys())
all_lengths = sorted(all_lengths)

fig, ax = plt.subplots(figsize=(11, 7))

for length in all_lengths:
    counts = [results[t]['length_counts'].get(length, 0) for t in thresholds]
    # нули заменяем на NaN, чтобы точки не рисовались на нуле
    counts_plot = [c if c > 0 else float('nan') for c in counts]

    ax.plot(thresholds, counts_plot, marker='o', markersize=9,
            linewidth=2, label=f'Длина {length}')

    for x, y in zip(thresholds, counts_plot):
        if y > 0:
            ax.annotate(f'{int(y)}', (x, y), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=10)

ax.set_xlabel('Порог поддержки')
ax.set_ylabel('Количество частых наборов (лог. шкала)')
ax.set_yscale('log')
ax.set_title('Количество частых наборов разной длины в зависимости от порога поддержки')

# --- Явные метки на оси Y ---
ax.yaxis.set_major_locator(
    mticker.FixedLocator([1, 2, 5, 10, 20, 50, 100, 200])
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))
ax.yaxis.set_minor_locator(
    mticker.LogLocator(base=10.0, subs='auto', numticks=100)
)
ax.yaxis.set_minor_formatter(mticker.NullFormatter())

ax.legend()
ax.grid(True, which='major', linestyle='--', alpha=0.7)
ax.grid(True, which='minor', linestyle=':', alpha=0.4)

plt.savefig('results/charts/itemsets_by_length_vs_support.png',
            dpi=300, bbox_inches='tight')
plt.show()