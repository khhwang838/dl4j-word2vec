package keichee.word2vec;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class App {
	
	private final static Logger log = LoggerFactory.getLogger(App.class);
	
	public static void main(String[] args) throws IOException {

		App app = new App();
		app.test(5);
		
	}
	
	private void test(int numOfWords) throws IOException {
		/**
		 * 데이터 불러오기
		 */
		log.info("Loading data....");
		ClassPathResource resource = new ClassPathResource("test_raw_sentences.txt");
		SentenceIterator iter = new LineSentenceIterator(resource.getFile());
		iter.setPreProcessor(new SentencePreProcessor() {
		    public String preProcess(String sentence) {
		    	
		    	// TODO : 형태소 분석하여 띄어쓰기 된 문장으로 변경
		    	
		    	
		    	return sentence.toLowerCase();
		    }
		});
		
		/**
		 * 데이터 토큰화 하기
		 */
		// Split on white spaces in the line to get words
		TokenizerFactory tokenizer = new DefaultTokenizerFactory();
		tokenizer.setTokenPreProcessor(new CommonPreprocessor());
		
		/**
		 * 모델 학습하기
		 */
		int batchSize = 1000;
//		int iterations = 3;
		int iterations = 3000;
		int layerSize = 150;

		log.info("Building model....");
		Word2Vec vec = new Word2Vec.Builder()
			.batchSize(batchSize) //# words per minibatch.
			.minWordFrequency(3) //
			.useAdaGrad(false) //
			.layerSize(layerSize) // word feature vector size
			.iterations(iterations) // # iterations to train
			.learningRate(0.025) //
			.minLearningRate(1e-3) // learning rate decays wrt # words. floor learning
			.negativeSample(10) // sample size 10 words
			.iterate(iter) //
			.tokenizerFactory(tokenizer)
			.build();
		vec.fit();
		
		
		/**
		 * 모델 학습 결과 평가하기
		 */
		// Write word vectors
		WordVectorSerializer.writeWordVectors(vec, "word_vectors.txt");

		/*
		 * 특정 단어와 관련도가 높은 단어 추출하기
		 */
//		Collection<String> lst = vec.wordsNearest("day", 10);
		Collection<String> lstNearest = vec.wordsNearest("쓰레기", numOfWords);
		log.info("Closest words of '쓰레기' : {}", lstNearest);

		/*
		 * 특정 단어와 정확도(유사도)(?)가 높은 단어 추출하기
		 */
		double accuracy = 0.7f;
		Collection<String> lstSimilar = vec.similarWordsInVocabTo("쓰레기", accuracy);
		log.info("Similar words of '쓰레기' by accuracy of {} : {}", accuracy, lstSimilar);
		
		/*
		 * 특정 단어와 가까운(?) 단어 추출하기
		 */
		Collection<String> kingList = vec.wordsNearest(Arrays.asList("음식물", "쓰레기"), Arrays.asList("queen", "woman"), numOfWords);
		log.info("Neareast words of '음식물, 쓰레기' not 'queen, woman' : {}", kingList);
//		Collection<String> rnnList = vec.wordsNearest(Arrays.asList("인공지능", "신경망"), Arrays.asList("rnn"), 10);
//		log.info("Neareast of word 'RNN' : {}", rnnList);

		/**
		 * 정확도 판단하기
		 */
		List<String> questions = Arrays.asList("쓰레기 king queen 봉투");
		Map<String, Double> accuracyResult = vec.accuracy(questions);
		log.info("accuracyResult: {}", accuracyResult);
		
		/**
		 * 학습 모델 저장
		 */
		log.info("Saving vectors....");
		WordVectorSerializer.writeFullModel(vec, "learn_model.txt");
		
		/**
		 * 벡터를 다시 메모리에 올리기
		 */
		Word2Vec word2Vec = WordVectorSerializer.loadFullModel("learn_model.txt");
		
		/**
		 * Word2Vec을 룩업 테이블로 사용하기
		 */
		WeightLookupTable weightLookupTable = word2Vec.lookupTable();
		Iterator<INDArray> vectors = weightLookupTable.vectors();
		INDArray wordVector = word2Vec.getWordVectorMatrix("myword");
		double[] wordVectorArr = word2Vec.getWordVector("myword");
		
	}
}
