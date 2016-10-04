/**
 * 
 */
package edu.uci.jforests.learning.boosting;

import java.util.Map.Entry;

import edu.uci.jforests.util.ConfigHolder;

/**
 * @author phong_nguyen
 *
 */
public class LambdaMartMFConfig extends LambdaMARTConfig {

	private final static String NB_FACTORS = "lambdamartmf.nb-factors";
	public int nb_factors = 10;
	
	private final static String MU_ONE = "lambdamartmf.mu1";
	public double mu1 = 0.5;
	
	public String userFeatures = null;
	private final static String USER_FEATURES = "lambdamartmf.user-features";
	
	public String itemFeatures = null;
	private final static String ITEM_FEATURES = "lambdamartmf.item-features";
	
	private final static String USER_SAMPLING_RATE = "lambdamartmf.user-sub-sampling";
	public double userSamplingRate = 1.0;
	
	private final static String ITEM_SAMPLING_RATE = "lambdamartmf.item-sub-sampling";
	public double itemSamplingRate = 1.0;
	
	private final static String DEBUG_FLAG = "lambdamartmf.debug";
	public boolean debug = false;
	
	@Override
	public void init(ConfigHolder config) {
		super.init(config);
		for (Entry<Object, Object> entry : config.getEntries()) {
			String key = ((String) entry.getKey()).toLowerCase();
			String value = ((String) entry.getValue()).trim();

			if (key.equals(NB_FACTORS)) {
				nb_factors = Integer.parseInt(value);
			} 
			else if (key.equals(MU_ONE)) {
				mu1 = Double.parseDouble(value);
			} 
			else if (key.equals(USER_FEATURES)) {
				userFeatures = value;
			} 
			else if (key.equals(ITEM_FEATURES)) {
				itemFeatures = value;
			} 
			else if (key.equals(USER_SAMPLING_RATE)) {
				userSamplingRate = Double.parseDouble(value);
			}
			else if (key.equals(ITEM_SAMPLING_RATE)) {
				itemSamplingRate = Double.parseDouble(value);
			}
			else if (key.equals(DEBUG_FLAG)) {
				debug = Boolean.parseBoolean(value);
			}
		}
	}
}
