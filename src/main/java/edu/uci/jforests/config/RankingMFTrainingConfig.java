/**
 * 
 */
package edu.uci.jforests.config;

import java.util.Map.Entry;

import edu.uci.jforests.util.ConfigHolder;

/**
 * @author phong_nguyen
 *
 */
public class RankingMFTrainingConfig extends RankingTrainingConfig {

	public String inputUserSimilarityFiles = null;
	private final static String RANKING_INPUT_USER_SIMILARITY_FILES = "ranking.input.user-similarity-files";
	
	public int inputUserNearestNeighbours = 10;
	private final static String RANKING_INPUT_USER_NEAREST_NEIGHBOURS = "ranking.input.user-nearest-neighbours";
	
	public double lambdaInputUser = 0;
	private final static String RANKING_LAMBDA_INPUT_USER = "ranking.input.user.lambda";
	
	public String outputUserSimilarityFiles = null;
	private final static String RANKING_OUTPUT_USER_SIMILARITY_FILES = "ranking.output.user-similarity-files";
	
	public int outputUserNearestNeighbours = 10;
	private final static String RANKING_OUTPUT_USER_NEAREST_NEIGHBOURS = "ranking.output.user-nearest-neighbours";

	public double lambdaOutputUser = 0;
	private final static String RANKING_LAMBDA_OUTPUT_USER = "ranking.output.user.lambda";
	
	public void init(ConfigHolder config) {
		super.init(config);
		for (Entry<Object, Object> entry : config.getEntries()) {
			String key = ((String) entry.getKey()).toLowerCase();
			String value = (String) entry.getValue();

			if (key.equals(RANKING_INPUT_USER_SIMILARITY_FILES)) {
				inputUserSimilarityFiles = value;
			} else if (key.equals(RANKING_INPUT_USER_NEAREST_NEIGHBOURS)) {
				inputUserNearestNeighbours = Integer.parseInt(value);
			}  else if (key.equals(RANKING_LAMBDA_INPUT_USER)) {
				lambdaInputUser = Double.parseDouble(value);
			} else if (key.equals(RANKING_OUTPUT_USER_SIMILARITY_FILES)) {
				outputUserSimilarityFiles = value;
			} else if (key.equals(RANKING_OUTPUT_USER_NEAREST_NEIGHBOURS)) {
				outputUserNearestNeighbours = Integer.parseInt(value);
			} else if (key.equals(RANKING_LAMBDA_OUTPUT_USER)) {
				lambdaOutputUser = Double.parseDouble(value);
			}
		}
	}

}
