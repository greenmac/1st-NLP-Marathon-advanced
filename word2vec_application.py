import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# 下載NLTK所需的資源
# nltk.download('punkt')
# nltk.download('stopwords')
# print('-'*5)

# 載入文本數據
text = "I like playing football. He enjoys playing basketball. Football is a popular sport. Basketball is a fast-paced game."

# 文本預處理
stop_words = set(stopwords.words('english'))
sentences = sent_tokenize(text)  # 將文本拆分成句子

corpus = []
for sentence in sentences:
    words = word_tokenize(sentence.lower())  # 將句子轉換為小寫單詞列表
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]  # 去除非字母數字字符和停用詞
    corpus.append(filtered_words)

print(f'corpus:{corpus}')
print('-'*5)

# 建立Word2Vec模型
model = Word2Vec(corpus, min_count=1)

# 獲取詞向量
vector = model.wv['football']
print("詞向量:", vector)
print('-'*5)

# 查找相似詞
similar_words = model.wv.most_similar('football')

print("相似詞:")
for word, similarity in similar_words:
    print(word, similarity)
