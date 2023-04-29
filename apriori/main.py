from itertools import combinations
from typing import Iterable

Dataset = list[frozenset]
SubsetFrequencies = dict[frozenset, int]
Subset = frozenset


def main():
    data = read_data('data.txt')
    result: SubsetFrequencies = apriori(data, 2, min_size=3)

    for frequent_items, count in result.items():
        print(
            set_to_string(frequent_items),
            ": ",
            count,
            sep='')

        rules_it = get_rules(data, count, frequent_items, min_confidence=0.5)
        rules = list(rules_it)
        rules.sort(key=lambda r: -r.confidence)

        for rule in rules:
            print(
                set_to_string(rule.antecedent),
                " -> ",
                set_to_string(rule.consequent),
                " (confidence: ",
                ("%1.2f" % rule.confidence),
                ")", sep='')

        print()


def set_to_string(s: Subset) -> str:
    return '^'.join(s)


def read_data(file_name: str) -> Dataset:
    with open(file_name, 'r') as file:
        return [frozenset(line.strip().split(',')) for line in file]


def apriori(data: Dataset, min_support: int, min_size: int) -> SubsetFrequencies:
    assert (min_size > 1)

    candidate_set: dict[object, int] = {}
    for row in data:
        for item in row:
            if item not in candidate_set:
                candidate_set[item] = 1
            else:
                candidate_set[item] += 1

    for key, value in list(candidate_set.items()):
        if value < min_support:
            del candidate_set[key]

    c1: SubsetFrequencies = {Subset([key]): value for key, value in candidate_set.items()}
    c0 = {}

    k = 2
    while k <= min_size:
        c0, c1 = c1, c0
        c1.clear()
        for key1 in c0.keys():
            for key2 in c0.keys():
                if key1 == key2:
                    continue

                intersection = key1.intersection(key2)
                if len(intersection) != k - 2:
                    continue

                union = key1.union(key2)
                if union in c1:
                    continue
                count = get_count(data, union)
                if count >= min_support:
                    c1[union] = count
        k += 1

    return c1


def get_count(data: Dataset, subset: Subset) -> int:
    count = 0
    for row in data:
        if subset.issubset(row):
            count += 1
    return count


class Rule:
    def __init__(self, antecedent: Subset, consequent: Subset, confidence: float):
        self.antecedent = antecedent
        self.consequent = consequent
        self.confidence = confidence


def get_rules(data: Dataset, count: int, frequent_set: Subset, min_confidence: float) -> "Iterable[Rule]":
    def non_empty_subsets(s: list) -> "Iterable[Subset]":
        for r in range(1, len(s)):
            for subset in combinations(s, r):
                yield Subset(subset)

    for itemset in non_empty_subsets(list(frequent_set)):
        antecedent = itemset
        count_antecedent = get_count(data, antecedent)
        confidence = count / count_antecedent

        if confidence >= min_confidence:
            consequent = frequent_set.difference(itemset)
            yield Rule(antecedent, consequent, confidence)


if __name__ == "__main__":
    main()
