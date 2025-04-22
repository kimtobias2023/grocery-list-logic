from gensim.models import KeyedVectors

# Load from .vec (can take a few minutes)
print("Loading FastText .vec file …")
vec_path = "models/crawl-300d-2M-subword.vec"
model = KeyedVectors.load_word2vec_format(vec_path, binary=False)

# Save in faster .kv format
kv_path = "models/fasttext.kv"
print(f"Saving to {kv_path} …")
model.save(kv_path)
print("✅ Done.")
