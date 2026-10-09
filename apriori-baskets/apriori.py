# apriori.py
"""
Реализация алгоритма Apriori для поиска частых наборов объектов.
"""

import time
from itertools import combinations
from collections import defaultdict
from typing import List, Set, Dict, Tuple, FrozenSet


def load_transactions(filepath: str) -> List[Set[str]]:
    """
    Загружает транзакции из CSV-файла.
    Каждая строка — корзина с товарами, разделёнными запятыми.
    Возвращает список множеств товаров.
    """
    transactions = []
    # Пробуем несколько кодировок по очереди
    for encoding in ('utf-8-sig', 'utf-8', 'cp1251', 'latin-1'):
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items = [item.strip() for item in line.split(',') if item.strip()]
                    if items:
                        transactions.append(set(items))
            print(f"Файл прочитан в кодировке: {encoding}")
            return transactions
        except UnicodeDecodeError:
            transactions = []   # сбросить и попробовать следующую
            continue
    raise RuntimeError("Не удалось определить кодировку файла")

def count_support(transactions: List[Set[str]], candidates: List[FrozenSet[str]]) -> Dict[FrozenSet[str], int]:
    """
    Подсчитывает абсолютную поддержку (число транзакций) для каждого кандидата.
    """
    counts = defaultdict(int)
    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                counts[candidate] += 1
    return counts


def generate_candidates(prev_frequent: List[FrozenSet[str]], k: int) -> List[FrozenSet[str]]:
    """
    Генерация кандидатов длины k на основе частых наборов длины k-1.
    Используется соединение (join) и отсечение (prune).
    """
    candidates = set()
    n = len(prev_frequent)

    # Шаг соединения: объединяем пары наборов, различающихся одним элементом
    for i in range(n):
        for j in range(i + 1, n):
            a = prev_frequent[i]
            b = prev_frequent[j]
            union = a | b
            if len(union) == k:
                # Шаг отсечения: все подмножества длины k-1 должны быть частыми
                subsets = combinations(union, k - 1)
                all_frequent = all(frozenset(s) in set(prev_frequent) for s in subsets)
                if all_frequent:
                    candidates.add(frozenset(union))
    return list(candidates)


def apriori(transactions: List[Set[str]],
            min_support: float = 0.01,
            sort_by: str = 'support') -> List[Tuple[FrozenSet[str], float, int]]:
    """
    Алгоритм Apriori.

    Параметры:
        transactions : список транзакций (каждая — множество товаров)
        min_support  : порог поддержки (доля от 0 до 1)
        sort_by      : 'support' — сортировка по убыванию поддержки,
                       'lex'     — лексикографическая сортировка

    Возвращает список кортежей (набор, относительная поддержка, абсолютная поддержка),
    отсортированный заданным способом.
    """
    n_transactions = len(transactions)
    min_count = max(1, int(min_support * n_transactions + 0.9999))  # ceil

    all_frequent = {}

    # ---------- Шаг 1: частые наборы длины 1 ----------
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1

    frequent_1 = [frozenset([item]) for item, cnt in item_counts.items() if cnt >= min_count]
    for itemset in frequent_1:
        all_frequent[itemset] = item_counts[next(iter(itemset))]

    prev_frequent = frequent_1
    k = 2

    # ---------- Итеративное построение наборов длины k ----------
    while prev_frequent:
        candidates = generate_candidates(prev_frequent, k)
        if not candidates:
            break
        counts = count_support(transactions, candidates)
        current_frequent = [c for c in candidates if counts[c] >= min_count]
        for itemset in current_frequent:
            all_frequent[itemset] = counts[itemset]

        prev_frequent = current_frequent
        k += 1

    # ---------- Формирование результата ----------
    result = []
    for itemset, cnt in all_frequent.items():
        result.append((itemset, cnt / n_transactions, cnt))

    if sort_by == 'lex':
        result.sort(key=lambda x: (sorted(x[0]), -x[2]))
    else:  # по убыванию поддержки
        result.sort(key=lambda x: (-x[1], sorted(x[0])))

    return result


def run_apriori_with_time(transactions: List[Set[str]],
                          min_support: float,
                          sort_by: str = 'support'):
    """
    Запускает Apriori и возвращает (результат, время выполнения в секундах).
    """
    start = time.perf_counter()
    result = apriori(transactions, min_support=min_support, sort_by=sort_by)
    elapsed = time.perf_counter() - start
    return result, elapsed