import pandas as pd
import numpy as np

def split(chunks=10):
    print("Let's do some DaTaScIeNcE")

    print("reading...")
    df = pd.read_csv("../../full_dataset.csv")

    print("df read. Shuffling...")
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("splitting...")
    split_dfs = np.array_split(df_shuffled, chunks)

    for i, split_df in enumerate(split_dfs):
        split_df = split_df.drop(split_df.columns[0], axis=1)
        split_df[['difficulty', 'time', 'cost', 'method']] = split_df.apply(heuristic_tags, axis=1)

        split_df.to_csv(f"dataset/recipies_dataset_tagged_chunk_{2 + i * 2}%.csv", index=True)
        print(f"saved {i}% of data set")

    print(df)


def heuristic_tags(row):
    tags = {}

    # Difficulty heuristic
    num_ingredients = len(row['ingredients'].split(','))
    tags['difficulty'] = 'Easy' if num_ingredients <= 7 else 'Hard'

    # Time heuristic
    slow_cooking_speed_tags = ["hour", "hrs", "hours", "overnight"]
    quick_cooking_speed_tags = ["minute", "minutes", "mins", "quick", "fast"]

    directions = row['directions'].lower()

    tags['time'] = 'Average'

    for quick_cooking_speed_tag in quick_cooking_speed_tags:
        if quick_cooking_speed_tag in directions:
            tags['time'] = 'Quick'
            break

    for slow_cooking_speed_tag in slow_cooking_speed_tags:
        if slow_cooking_speed_tag in directions:
            tags['time'] = 'Slow'
            break

    # Cost heuristic
    # Items are arbitrary assigned to be expensive.
    expensive_items = ['lobster', 'salmon', 'truffle', 'steak', 'beef', 'vine', 'vinegar', 'kiwi', 'oyster']
    if any(item in row['ingredients'].lower() for item in expensive_items):
        tags['cost'] = 'Expensive'
    else:
        tags['cost'] = 'Cheap'

    # Method heuristic
    methods = ['fried', 'baked', 'boiled', 'raw', 'grilled', 'roasted', 'marinated'
               'fry', 'bake', 'boil', 'grill', 'broil']
    tags['method'] = 'Other'
    for method in methods:
        if method in directions:
            tags['method'] = method.capitalize()
            break

    return pd.Series(tags)


if __name__ == '__main__':
    split(chunks=50)
