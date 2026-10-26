"""
Quick subsample: take the most active 500 users from MovieLens 1M
to create a small training dataset that can train in <5 minutes on CPU.
"""
import os, json, pickle
import numpy as np

SRC = os.path.join("datasets", "movielens")
DST = os.path.join("datasets", "movielens_small")

def main():
    with open(os.path.join(SRC, "session_train.pkl"), "rb") as f:
        train = pickle.load(f)
    with open(os.path.join(SRC, "session_test.pkl"), "rb") as f:
        test = pickle.load(f)
    with open(os.path.join(SRC, "item_catgy.pkl"), "rb") as f:
        item_catgy = pickle.load(f)
    with open(os.path.join(SRC, "item_category.pkl"), "rb") as f:
        item_category = pickle.load(f)
    with open(os.path.join(SRC, "session_info.json")) as f:
        session_info = json.load(f)

    # Subsample: take first 5000 train sessions and 1000 test sessions
    N_TRAIN = 5000
    N_TEST  = 1000
    train_small = train[:N_TRAIN]
    test_small  = test[:N_TEST]

    session_info_small = dict(session_info)
    session_info_small["Total number of train session"] = len(train_small)
    session_info_small["Total number of test session"]  = len(test_small)
    session_info_small["beizhu"] = "movielens-small-demo"

    os.makedirs(DST, exist_ok=True)
    with open(os.path.join(DST, "session_train.pkl"), "wb") as f:
        pickle.dump(train_small, f)
    with open(os.path.join(DST, "session_test.pkl"), "wb") as f:
        pickle.dump(test_small, f)
    with open(os.path.join(DST, "item_catgy.pkl"), "wb") as f:
        pickle.dump(item_catgy, f)
    with open(os.path.join(DST, "item_category.pkl"), "wb") as f:
        pickle.dump(item_category, f)
    with open(os.path.join(DST, "session_info.json"), "w") as f:
        json.dump(session_info_small, f, indent=2)

    print(f"Small dataset saved to {DST}")
    print(f"  Train: {len(train_small)} sessions")
    print(f"  Test:  {len(test_small)} sessions")
    print(f"  Items: {session_info_small['num_of_item[not include 0]']}")
    print(f"  Cats:  {session_info_small['num_of_category']}")

if __name__ == "__main__":
    main()
