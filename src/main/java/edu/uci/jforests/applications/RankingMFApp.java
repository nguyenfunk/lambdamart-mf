/**
 * 
 */
package edu.uci.jforests.applications;

import edu.uci.jforests.config.RankingMFTrainingConfig;
import edu.uci.jforests.config.RankingTrainingConfig;
import edu.uci.jforests.dataset.Dataset;
import edu.uci.jforests.dataset.RankingDataset;
import edu.uci.jforests.learning.LearningModule;
import edu.uci.jforests.learning.boosting.LambdaMartMF;
import edu.uci.jforests.sample.RankingMFSample;
import edu.uci.jforests.sample.Sample;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

/**
 * @author phong_nguyen
 *
 */
public class RankingMFApp extends RankingApp {

	protected TIntObjectHashMap<TIntDoubleHashMap> inputUserSims, outputUserSims;
	protected double lambdaInputUser, lambdaOutputUser;
	
	/**
	 * 
	 */
	public RankingMFApp() {
		super();
	}

	@Override
	protected LearningModule getLearningModule(String name) throws Exception {
		int maxTrainInstances = ((RankingDataset)trainDataset).numInstances;
		if (name.equals("LambdaMartMF")) {
			LambdaMartMF learner = new LambdaMartMF(inputItemSims, outputItemSims, lambdaInputItem, lambdaOutputItem, inputUserSims, outputUserSims, lambdaInputUser, lambdaOutputUser);
			learner.init(configHolder, (RankingDataset) trainDataset, maxTrainInstances, (validDataset != null ? validDataset.numInstances
					: trainDataset.numInstances), evaluationMetric);
			return learner;
		}
		else {
			return super.getLearningModule(name);
		}
	}
	
	@Override
	protected Sample createSample(Dataset dataset, boolean trainSample) {
		RankingMFSample sample = new RankingMFSample((RankingDataset) dataset);
		return sample;
	}
	
	@Override
	protected void loadConfig() {
		trainingConfig = new RankingMFTrainingConfig();
		trainingConfig.init(configHolder);
	}
	
	@Override
	protected void init() throws Exception {
		super.init();
	
		String inputUserSimilarityFiles = ((RankingMFTrainingConfig) trainingConfig).inputUserSimilarityFiles; 
		if(inputUserSimilarityFiles != null) this.inputUserSims = getSimilarities(inputUserSimilarityFiles.split(" "), ((RankingMFTrainingConfig) trainingConfig).inputUserNearestNeighbours);
		
		String outputUserSimilarityFiles = ((RankingMFTrainingConfig) trainingConfig).outputUserSimilarityFiles; 
		if(outputUserSimilarityFiles != null) this.outputUserSims = getSimilarities(outputUserSimilarityFiles.split(" "), ((RankingMFTrainingConfig) trainingConfig).outputUserNearestNeighbours);
		
		lambdaInputUser = ((RankingMFTrainingConfig) trainingConfig).lambdaInputUser;
		lambdaOutputUser = ((RankingMFTrainingConfig) trainingConfig).lambdaOutputUser;
	}
}
