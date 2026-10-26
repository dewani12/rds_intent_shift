"""
MovieLens 1M → CoDeR dataset preprocessor
Downloads ML-1M, maps genres to category IDs, and creates the
required session_train.pkl / session_test.pkl / item_catgy.pkl /
item_category.pkl / session_info.json files inside datasets/movielens/.
"""

import os
import io
import ssl
import json
import pickle
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────────────────────────────
URL       = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
RAW_DIR   = "raw_data"
SAVE_PATH = os.path.join("datasets", "movielens")

MIN_OCCURRENCES    = 5    # min interactions per item/category
MIN_SESSION_LEN    = 2    # min items in a session
MAX_SESSION_LEN    = 50
TRAIN_SIZE         = 0.8
SESSION_ENHANCEMENT = True   # augment by sliding window
# ── Helpers ──────────────────────────────────────────────────────────────────

def download_ml1m():
    zip_path = os.path.join(RAW_DIR, "ml-1m.zip")
    os.makedirs(RAW_DIR, exist_ok=True)
    if not os.path.exists(zip_path):
        print("Downloading MovieLens 1M ...")
        # Bypass SSL verification (common issue on Windows with Python 3.14)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(URL, context=ctx) as r, open(zip_path, 'wb') as f:
            total = int(r.headers.get('Content-Length', 0))
            downloaded = 0
            chunk = 65536
            while True:
                data = r.read(chunk)
                if not data:
                    break
                f.write(data)
                downloaded += len(data)
                if total:
                    print(f"  {downloaded/1e6:.1f} / {total/1e6:.1f} MB", end='\r')
        print("\nDownload complete.")
    else:
        print("Archive already present, skipping download.")
    return zip_path


def load_ratings(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("ml-1m/ratings.dat") as f:
            ratings = pd.read_csv(
                f, sep="::", engine="python",
                names=["user_id", "item_id", "rating", "timestamp"],
                encoding="latin-1"
            )
        with zf.open("ml-1m/movies.dat") as f:
            movies = pd.read_csv(
                f, sep="::", engine="python",
                names=["item_id", "title", "genres"],
                encoding="latin-1"
            )
    return ratings, movies


def build_dataframe(ratings, movies):
    # Use the *first* listed genre as the item's category
    movies["category"] = movies["genres"].str.split("|").str[0]
    df = ratings.merge(movies[["item_id", "category"]], on="item_id")
    df = df.rename(columns={"category": "category_id"})
    df = df[["user_id", "item_id", "category_id", "timestamp"]]
    # Keep only ratings >= 4 (implicit positive feedback)
    df = df[ratings["rating"] >= 4].copy()
    df = df.reset_index(drop=True)
    print(f"Interactions after rating filter: {len(df):,}")
    return df


# ── Session building ──────────────────────────────────────────────────────────

def filter_items(df, min_occ):
    item_cnt  = df["item_id"].value_counts()
    catgy_cnt = df["category_id"].value_counts()
    valid_items  = item_cnt[item_cnt  > min_occ].index
    valid_catgy  = catgy_cnt[catgy_cnt > min_occ].index
    df = df[df["item_id"].isin(valid_items) & df["category_id"].isin(valid_catgy)]
    return df.reset_index(drop=True)


def build_sessions(df, min_len, max_len, sess_enhancement):
    """Group by user, sort by timestamp; each day = one session."""
    sessions = []
    df = df.sort_values(["user_id", "timestamp"])
    # Treat each user's sorted history as a single session sequence
    for user_id, grp in tqdm(df.groupby("user_id"), desc="Building sessions"):
        items  = grp["item_id"].tolist()
        cats   = grp["category_id"].tolist()
        times  = grp["timestamp"].tolist()

        if len(items) < min_len:
            continue
        if len(items) > max_len + 1:
            items = items[:max_len + 1]
            cats  = cats[:max_len + 1]
            times = times[:max_len + 1]

        start_time = str(times[0])
        end_time   = str(times[-1])

        if sess_enhancement:
            for i in range(1, len(items) + 2 - min_len):
                prefix_items = items[:-i]
                prefix_cats  = cats[:-i]
                target_item  = items[-i]
                target_cat   = cats[-i]
                if len(prefix_items) < 1:
                    continue
                sessions.append(
                    (user_id, start_time, end_time,
                     prefix_items, prefix_cats,
                     target_item, target_cat)
                )
        else:
            sessions.append(
                (user_id, start_time, end_time,
                 items[:-1], cats[:-1], items[-1], cats[-1])
            )
    return sessions


def split_train_test(sessions, train_size):
    df = pd.DataFrame(
        sessions,
        columns=["user_id", "start_time", "end_time",
                 "item_list", "category_list", "target_item", "target_category"]
    )
    df = df.sort_values("start_time").reset_index(drop=True)
    k  = int(len(df) * train_size)
    train = df.iloc[:k].values.tolist()
    test  = df.iloc[k:].values.tolist()
    return train, test


def save_data(save_path, session_info, category, train_list, test_list, item_category):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "session_info.json"), "w") as f:
        json.dump(session_info, f, indent=2)
    with open(os.path.join(save_path, "item_catgy.pkl"), "wb") as f:
        pickle.dump(category, f)
    with open(os.path.join(save_path, "session_train.pkl"), "wb") as f:
        pickle.dump(train_list, f)
    with open(os.path.join(save_path, "session_test.pkl"), "wb") as f:
        pickle.dump(test_list, f)
    with open(os.path.join(save_path, "item_category.pkl"), "wb") as f:
        pickle.dump(item_category, f)
    print(f"\nSaved dataset to: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    zip_path = download_ml1m()
    ratings, movies = load_ratings(zip_path)
    df = build_dataframe(ratings, movies)

    # ── Encode ──
    df = filter_items(df, MIN_OCCURRENCES)
    print(f"Interactions after frequency filter: {len(df):,}")

    item_enc  = LabelEncoder()
    catgy_enc = LabelEncoder()
    user_enc  = LabelEncoder()

    df["item_id"]     = item_enc.fit_transform(df["item_id"])     + 1
    df["category_id"] = catgy_enc.fit_transform(df["category_id"]) + 1
    df["user_id"]     = user_enc.fit_transform(df["user_id"])

    n_items    = len(item_enc.classes_)
    n_cats     = len(catgy_enc.classes_)
    n_users    = len(user_enc.classes_)

    print(f"Items: {n_items}  |  Categories: {n_cats}  |  Users: {n_users}")

    # item_catgy: list of length n_items; index i → category of item (i+1)
    item_cat_df = (
        df.drop_duplicates("item_id")
          .sort_values("item_id")
          .set_index("item_id")["category_id"]
    )
    # Ensure full coverage 1..n_items
    category = [int(item_cat_df.get(i, 1)) for i in range(1, n_items + 1)]

    # item_category dict: {item_id: [category_id]}  (used in model deconfounder)
    item_category = {
        int(row.item_id): [int(row.category_id)]
        for _, row in df.drop_duplicates("item_id").iterrows()
    }

    sessions = build_sessions(df, MIN_SESSION_LEN, MAX_SESSION_LEN, SESSION_ENHANCEMENT)
    print(f"Total sessions (after augmentation): {len(sessions):,}")

    train_list, test_list = split_train_test(sessions, TRAIN_SIZE)
    print(f"Train: {len(train_list):,}  |  Test: {len(test_list):,}")

    session_info = {
        "dataset": "movielens-1m",
        "num_of_user": n_users,
        "num_of_item[not include 0]": n_items,
        "num_of_category": n_cats,
        "min_session_len": MIN_SESSION_LEN,
        "max_session_len": MAX_SESSION_LEN,
        "min_occurrences": MIN_OCCURRENCES,
        "train_size": TRAIN_SIZE,
        "Total number of session after enhancement": len(sessions),
        "Total number of train session": len(train_list),
        "Total number of test session": len(test_list),
    }

    save_data(SAVE_PATH, session_info, category, train_list, test_list, item_category)
    print("\n✅ Preprocessing complete! Ready to train.")
    print(f"   Run:  python main.py --dataset movielens --epoch 10 --batch_size 64")


if __name__ == "__main__":
    main()
