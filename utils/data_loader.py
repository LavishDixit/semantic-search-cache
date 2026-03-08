import os

def load_dataset(base_path="data/mini_newsgroups"):

    texts = []
    labels = []

    for category in os.listdir(base_path):

        category_path = os.path.join(base_path, category)

        if not os.path.isdir(category_path):
            continue

        for file in os.listdir(category_path):

            file_path = os.path.join(category_path, file)

            try:
                with open(file_path, "r", errors="ignore") as f:
                    texts.append(f.read())
                    labels.append(category)
            except:
                continue

    return texts, labels