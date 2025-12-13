from pyspark import SparkContext, SparkConf
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk

nltk.download('stopwords')

def main():
    conf = SparkConf().setAppName("TextAnalysis").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    
    stop_words = set(stopwords.words('russian'))
    stemmer = SnowballStemmer("russian")
    
    def clean_text(line):
        import re
        line = line.lower()
        line = re.sub(r'[^а-яё\s]', '', line)
        words = line.split()
        return [word for word in words 
                if word not in stop_words and len(word) > 2]
    
    text_rdd = sc.textFile("text.txt")
    words_rdd = text_rdd.flatMap(clean_text)
    
    word_count = words_rdd.map(lambda word: (word, 1)) \
                         .reduceByKey(lambda a, b: a + b)
    
    top_50 = word_count.takeOrdered(50, key=lambda x: -x[1])
    print("Top50 самых частых слов:")
    print(", ".join(f"{word}: {count}" for word, count in top_50))
    
    bottom_50 = word_count.takeOrdered(50, key=lambda x: x[1])
    print("\nTop50 самых редких слов:")
    print(", ".join(f"{word}: {count}" for word, count in bottom_50))
    
    # Стемминг
    stemmed_rdd = words_rdd.map(lambda word: stemmer.stem(word))
    stemmed_count = stemmed_rdd.map(lambda word: (word, 1)) \
                              .reduceByKey(lambda a, b: a + b)
    
    top_50_stemmed = stemmed_count.takeOrdered(50, key=lambda x: -x[1])
    print("\nTop50 самых частых слов после стемминга:")
    print(", ".join(f"{word}: {count}" for word, count in top_50_stemmed))
    
    bottom_50_stemmed = stemmed_count.takeOrdered(50, key=lambda x: x[1])
    print("\nTop50 самых редких слов после стемминга:")
    print(", ".join(f"{word}: {count}" for word, count in bottom_50_stemmed))
    
    sc.stop()

if __name__ == "__main__":
    main()