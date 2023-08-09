from gensim.models import Word2Vec

sentences = [
    ['i', 'love', 'machine', 'learning'],
    ['machine', 'learning', 'is', 'fun'],
    ['deep', 'learning', 'is', 'a', 'subfield', 'of', 'AI']
]

# 訓練Word2Vec模型
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)  # 使用Skip-gram模型

# 保存訓練好的模型
model.save("word2vec_model")

# 加載模型
loaded_model = Word2Vec.load("word2vec_model")

# 查詢某個詞彙的詞向量
word_vector = loaded_model.wv['learning']
print("Vector representation of 'machine':", word_vector)
