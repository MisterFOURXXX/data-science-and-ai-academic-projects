import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import chromadb
from chromadb.config import Settings
from pathlib import Path

def visualize_vectorstore(config):
    chroma_client = chromadb.PersistentClient(
        path=config.vectorstore.chroma_persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection = chroma_client.get_collection("stackoverflow_coding_train")
    
    all_data = collection.get(include=["embeddings", "documents", "metadatas"])
    
    embeddings = np.array(all_data["embeddings"])
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    
    print(f"Embedding matrix shape: {embeddings.shape}")
    
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d_pca = pca.fit_transform(embeddings)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    reducer = umap.UMAP(n_neighbors=15, n_components=2, metric="cosine", min_dist=0.1, random_state=42, verbose=False)
    embeddings_2d_umap = reducer.fit_transform(embeddings)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], s=10, alpha=0.6, c='royalblue')
    axes[0, 0].set_title("PCA Projection - All Embeddings")
    axes[0, 0].set_xlabel("PC1")
    axes[0, 0].set_ylabel("PC2")
    axes[0, 0].grid(True, alpha=0.3)
    
    scores = [m.get('score', 0) for m in metadatas]
    scatter = axes[0, 1].scatter(embeddings_2d_umap[:, 0], embeddings_2d_umap[:, 1], c=scores, cmap='viridis', s=10, alpha=0.6)
    axes[0, 1].set_title("UMAP Colored by Answer Score")
    axes[0, 1].set_xlabel("UMAP1")
    axes[0, 1].set_ylabel("UMAP2")
    plt.colorbar(scatter, ax=axes[0, 1], label="Score")
    
    lengths = [m.get('length', 0) for m in metadatas]
    scatter2 = axes[1, 0].scatter(embeddings_2d_umap[:, 0], embeddings_2d_umap[:, 1], c=lengths, cmap='plasma', s=10, alpha=0.6)
    axes[1, 0].set_title("UMAP Colored by Answer Length")
    axes[1, 0].set_xlabel("UMAP1")
    axes[1, 0].set_ylabel("UMAP2")
    plt.colorbar(scatter2, ax=axes[1, 0], label="Length (chars)")
    
    code_flags = [m.get('has_code', 0) for m in metadatas]
    axes[1, 1].scatter(embeddings_2d_umap[:, 0], embeddings_2d_umap[:, 1], c=code_flags, cmap='coolwarm', s=10, alpha=0.6)
    axes[1, 1].set_title("UMAP Colored by Code Presence")
    axes[1, 1].set_xlabel("UMAP1")
    axes[1, 1].set_ylabel("UMAP2")
    
    plt.tight_layout()
    plt.savefig("metrics/vectorstore_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    reducer_3d = umap.UMAP(n_neighbors=15, n_components=3, metric="cosine", min_dist=0.1, random_state=42, verbose=False)
    embeddings_3d = reducer_3d.fit_transform(embeddings)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter3d = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=scores, cmap='viridis', s=8, alpha=0.6)
    ax.set_title("UMAP 3D Projection - Semantic Space")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_zlabel("UMAP3")
    plt.colorbar(scatter3d, ax=ax, label="Score")
    plt.savefig("metrics/vectorstore_3d_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualization completed")

if __name__ == "__main__":
    from config import config
    visualize_vectorstore(config)