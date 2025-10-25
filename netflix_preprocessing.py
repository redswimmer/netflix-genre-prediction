"""
Netflix Genre Prediction - Complete Preprocessing Pipeline
Uses Ollama embeddinggemma for text embeddings
"""

import json
import pickle
from pathlib import Path

import numpy as np
import ollama
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from tqdm import tqdm


class NetflixPreprocessor:
    """Complete preprocessing pipeline for Netflix genre prediction"""

    def __init__(self, embedding_model="embeddinggemma", embedding_dim=768):
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.mlb = MultiLabelBinarizer()  # For multi-label genres
        self.cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.genre_names = None
        self.feature_names = None

    def load_data(self, filepath):
        """Load Netflix CSV data"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        return df

    def parse_genres(self, df):
        """Parse comma-separated genres into lists"""
        print("\nParsing genres...")

        # Split genres by comma and strip whitespace
        df["genre_list"] = df["genres"].apply(
            lambda x: [g.strip() for g in x.split(",")] if pd.notna(x) else []
        )

        # Statistics
        genre_counts = df["genre_list"].apply(len)
        print(f"Average genres per item: {genre_counts.mean():.2f}")
        print(f"Min genres: {genre_counts.min()}, Max genres: {genre_counts.max()}")

        # Count unique genres
        all_genres = [g for genres in df["genre_list"] for g in genres]
        unique_genres = set(all_genres)
        print(f"Total unique genres: {len(unique_genres)}")

        return df

    def encode_genres(self, df):
        """Convert genre lists to multi-label binary matrix"""
        print("\nEncoding genres as multi-label targets...")

        # Fit and transform genres
        y = self.mlb.fit_transform(df["genre_list"])
        self.genre_names = self.mlb.classes_

        print(f"Genre matrix shape: {y.shape}")
        print(f"Genre labels: {len(self.genre_names)}")
        print(f"Top 10 genres: {list(self.genre_names[:10])}")

        return y

    def generate_embeddings(self, texts, batch_size=50, save_path=None):
        """
        Generate embeddings using Ollama (or load from cache if exists)

        Args:
            texts: List of text strings
            batch_size: Process in batches for progress tracking
            save_path: Optional path to save/load embeddings
        """
        # Check if embeddings already exist
        if save_path and Path(save_path).exists():
            print(f"\n✓ Found existing embeddings at {save_path}")
            embeddings = np.load(save_path)
            print(f"Loaded embeddings shape: {embeddings.shape}")

            # Verify shape matches
            if (
                embeddings.shape[0] == len(texts)
                and embeddings.shape[1] == self.embedding_dim
            ):
                print("✓ Embeddings match expected dimensions, using cached version")
                return embeddings
            else:
                print(
                    f"⚠ Shape mismatch (expected {len(texts)}x{self.embedding_dim}, got {embeddings.shape})"
                )
                print("Regenerating embeddings...")

        print(f"\nGenerating embeddings using {self.embedding_model}...")
        print(f"Processing {len(texts)} descriptions...")

        embeddings = []

        # Process with progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i : i + batch_size]

            for text in batch:
                try:
                    response = ollama.embed(
                        model=self.embedding_model,
                        input=str(text),  # Ensure it's a string
                    )
                    embeddings.append(response["embeddings"][0])
                except Exception as e:
                    print(f"\nError embedding text: {text[:50]}... Error: {e}")
                    # Use zero vector as fallback
                    embeddings.append([0.0] * self.embedding_dim)

        embeddings = np.array(embeddings)
        print(f"Embedding matrix shape: {embeddings.shape}")

        # Save embeddings if path provided
        if save_path:
            np.save(save_path, embeddings)
            print(f"Saved embeddings to {save_path}")

        return embeddings

    def encode_categorical_features(self, df):
        """Encode type and rating as categorical features"""
        print("\nEncoding categorical features (type, rating)...")

        # Handle missing values
        df["type_clean"] = df["type"].fillna("Unknown")
        df["rating_clean"] = df["rating"].fillna("Unknown")

        # One-hot encode
        cat_features = self.cat_encoder.fit_transform(
            df[["type_clean", "rating_clean"]]
        )

        # Get feature names (use actual column names from fitting)
        cat_names = self.cat_encoder.get_feature_names_out()

        print(f"Categorical features shape: {cat_features.shape}")
        print(f"Features: {list(cat_names)}")

        return cat_features, cat_names

    def combine_features(self, embeddings, categorical_features, cat_names):
        """Combine embeddings with categorical features"""
        print("\nCombining all features...")

        X = np.hstack([embeddings, categorical_features])

        # Create feature names
        embedding_names = [f"emb_{i}" for i in range(embeddings.shape[1])]
        self.feature_names = embedding_names + list(cat_names)

        print(f"Final feature matrix shape: {X.shape}")
        print(f"  - Embeddings: {embeddings.shape[1]} dimensions")
        print(f"  - Categorical: {categorical_features.shape[1]} dimensions")
        print(f"  - Total: {X.shape[1]} dimensions")

        return X

    def preprocess(self, csv_path, save_dir="processed_data"):
        """
        Complete preprocessing pipeline

        Args:
            csv_path: Path to Netflix CSV
            save_dir: Directory to save processed data

        Returns:
            X: Feature matrix
            y: Multi-label target matrix
            df: Original dataframe with additions
        """
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        # Load data
        df = self.load_data(csv_path)

        # Parse and encode genres
        df = self.parse_genres(df)
        y = self.encode_genres(df)

        # Generate text embeddings
        embeddings_path = save_dir / "embeddings.npy"
        embeddings = self.generate_embeddings(
            df["description"].tolist(), save_path=embeddings_path
        )

        # Encode categorical features
        categorical_features, cat_names = self.encode_categorical_features(df)

        # Combine all features
        X = self.combine_features(embeddings, categorical_features, cat_names)

        # Save processed data
        print(f"\nSaving processed data to {save_dir}...")
        np.save(save_dir / "X.npy", X)
        np.save(save_dir / "y.npy", y)
        df.to_csv(save_dir / "processed_df.csv", index=False)

        # Save preprocessor configuration
        config = {
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "genre_names": self.genre_names.tolist(),
            "feature_names": self.feature_names,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_genres": y.shape[1],
        }

        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save sklearn objects
        with open(save_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump({"mlb": self.mlb, "cat_encoder": self.cat_encoder}, f)

        print("\n✅ Preprocessing complete!")
        print(f"Files saved in: {save_dir}/")
        print(f"  - X.npy: Feature matrix {X.shape}")
        print(f"  - y.npy: Target matrix {y.shape}")
        print("  - embeddings.npy: Text embeddings")
        print("  - processed_df.csv: Full dataframe")
        print("  - config.json: Configuration")
        print("  - preprocessor.pkl: Sklearn objects")

        return X, y, df

    def create_train_test_split(
        self, X, y, test_size=0.2, random_state=42, save_dir="processed_data"
    ):
        """Create and save train/test splits"""
        print(f"\nCreating train/test split (test_size={test_size})...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        save_dir = Path(save_dir)
        np.save(save_dir / "X_train.npy", X_train)
        np.save(save_dir / "X_test.npy", X_test)
        np.save(save_dir / "y_train.npy", y_train)
        np.save(save_dir / "y_test.npy", y_test)

        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Saved splits to {save_dir}/")

        return X_train, X_test, y_train, y_test


def load_processed_data(data_dir="processed_data"):
    """Load previously processed data"""
    data_dir = Path(data_dir)

    print(f"Loading processed data from {data_dir}...")

    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")

    with open(data_dir / "config.json") as f:
        config = json.load(f)

    with open(data_dir / "preprocessor.pkl", "rb") as f:
        sklearn_objects = pickle.load(f)

    print(f"Loaded X: {X.shape}, y: {y.shape}")
    print(f"Genres: {len(config['genre_names'])}")

    return X, y, config, sklearn_objects


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = NetflixPreprocessor(
        embedding_model="embeddinggemma", embedding_dim=768
    )

    # Run full preprocessing pipeline
    X, y, df = preprocessor.preprocess(
        csv_path="NetFlix.csv", save_dir="processed_data"
    )

    # Create train/test splits
    X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(
        X, y, test_size=0.2, save_dir="processed_data"
    )

    # Display summary statistics
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total samples: {X.shape[0]}")
    print(f"Total features: {X.shape[1]}")
    print(f"Total genres: {y.shape[1]}")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"\nGenres: {preprocessor.genre_names[:10]}... (showing first 10)")
    print("\n✅ Ready for model training!")
