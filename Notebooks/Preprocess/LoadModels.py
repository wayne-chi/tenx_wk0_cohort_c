
import json
import joblib
import gensim
from yake import KeywordExtractor


def lda_display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
    return topics

class TfidfExtractor:
    def __init__(self, path, top_n=10):
        # Load the vectorizer from the specified path
        self.vectorizer = joblib.load(path)
        self.top_n = top_n
        # Extract feature names from the vectorizer
        self.feature_names = self.vectorizer.get_feature_names_out()

    def extract_keywords(self, new_text):
        # Transform the new text using the loaded vectorizer
        tfidf_matrix = self.vectorizer.transform([new_text])
        # Get the indices and data (tf-idf scores) from the first (and only) row of the matrix
        indices = tfidf_matrix[0].indices
        data = tfidf_matrix[0].data
        # Sort indices by the tf-idf scores in descending order
        sorted_indices = indices[data.argsort()[::-1]]
        # Return the top n keywords
        return [self.feature_names[i] for i in sorted_indices[:self.top_n]]

# Example usage:
# extractor = TfidfExtractor('path/to/vectorizer.pkl', top_n=5)
# keywords = extractor.extract_keywords('sample text to analyze')
# print(keywords)


class YAKEExtractor:
    def __init__(self, params_path, top=10):
        # Load YAKE parameters from the specified file
        with open(params_path, 'r') as f:
            self.yake_params = json.load(f)
        
        # Update the 'top' parameter
        self.yake_params['top'] = top
        
        # Initialize the YAKE keyword extractor
        self.extractor = KeywordExtractor(**self.yake_params)
    
    def extract_keywords(self, text):
        # Extract keywords from the provided text
        keywords = self.extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]




class TopicModelPredictor:
    def __init__(self, model_path, vectorizer_path, model_name, no_top_words=10):
        self.model_name = model_name.lower()
        self.no_top_words = no_top_words

        # Load the model based on the model name
        if self.model_name == 'lda':
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.feature_names = self.vectorizer.get_feature_names_out()
            self.topics = self._display_topics(self.model, self.feature_names, self.no_top_words)
        elif self.model_name == 'lsi':
            self.model = gensim.models.LsiModel.load(model_path)
            self.dictionary = gensim.corpora.Dictionary.load(vectorizer_path)
            self.topics = self.model.print_topics(num_words=self.no_top_words)
        elif self.model_name == 'nmf':
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.feature_names = self.vectorizer.get_feature_names_out()
            self.topics = self._display_nmf_topics(self.model, self.feature_names, self.no_top_words)
        else:
            raise ValueError("Model name must be 'lda', 'lsi', or 'nmf'")

    def _display_topics(self, model, feature_names, no_top_words):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            topics.append([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        return topics

    def _display_nmf_topics(self, model, feature_names, no_top_words):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            topics.append([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        return topics

    def transform(self, text):
        if self.model_name == 'lda' or self.model_name == 'nmf':
            return self.vectorizer.transform([text])
        elif self.model_name == 'lsi':
            processed_text = gensim.utils.simple_preprocess(text)
            return self.dictionary.doc2bow(processed_text)
        else:
            raise ValueError("Invalid model name for transformation")

    def predict_proba(self, text):
        transformed_text = self.transform(text)

        if self.model_name == 'lda' or self.model_name == 'nmf':
            return self.model.transform(transformed_text)
        elif self.model_name == 'lsi':
            return self.model[transformed_text]
        else:
            raise ValueError("Invalid model name for prediction")

    def predict(self, text):
        # Get the topic probabilities or scores
        topic_probas = self.predict_proba(text)

        if self.model_name == 'lda' or self.model_name == 'nmf':
            # Get the topic with the highest probability
            topic_idx = topic_probas.argmax(axis=1)[0]
            # Return the top words in the most relevant topic
            return ', '.join(self.topics[topic_idx])
        elif self.model_name == 'lsi':
            # Sort topics by score and get the most relevant one
            sorted_topics = sorted(topic_probas, key=lambda x: -x[1])
            most_relevant_topic = sorted_topics[0][0]
            return self.topics[most_relevant_topic][1]
        else:
            raise ValueError("Invalid model name for prediction")

# Example usage:
# predictor = TopicModelPredictor('path/to/model', 'path/to/vectorizer_or_dictionary', 'lda')
# readable_topics = predictor.predict("Your new document text here")
# print(readable_topics)
