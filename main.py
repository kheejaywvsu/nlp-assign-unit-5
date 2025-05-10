from utils.fetch_data import fetch_wikipedia_docs
from term_document_matrix import build_matrices
from cosine_similarity import compute_similarity
from word2vec_classifier import train_classifier

# Configuration
TOPICS = [
    "Python (programming language)",
    "JavaScript",
    "Data science",
    "Machine learning",
    "Deep learning"
]

def main():
    print("🚀 Starting Document Analysis Pipeline\n")

    try:
        # Phase 1: Data Acquisition & Preprocessing
        print("🔍 Fetching Wikipedia documents...")
        documents, titles = fetch_wikipedia_docs(TOPICS)
        print(f"✅ Retrieved {len(documents)} documents\n")

        # Phase 2: Term-Document Analysis
        print("📊 Building term-document matrices...")
        raw_df, tfidf_df = build_matrices(documents, titles)

        # Phase 3: Similarity Analysis
        print("\n📐 Calculating document similarities...")
        similarity_df, (doc1, doc2, score) = compute_similarity(tfidf_df.values, titles)

        # Phase 4: Embedding & Classification
        print("\n🤖 Training Word2Vec classifier...")
        accuracy = train_classifier(documents, titles)

        # Final Results
        print("\n📊 Pipeline Results:")
        print(f"➤ Most Similar Documents: '{doc1}' & '{doc2}' (Score: {score:.2f})")
        print(f"➤ Classification Accuracy: {accuracy:.2f}")

    except Exception as e:
        print(f"\n❌ Pipeline Failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    print("\n🎉 Analysis Complete!")
