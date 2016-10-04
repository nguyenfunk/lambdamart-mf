/**
 * 
 */
package edu.uci.jforests.learning.trees.regression;

import edu.uci.jforests.dataset.Feature;
import edu.uci.jforests.dataset.Histogram;
import edu.uci.jforests.learning.trees.CandidateSplitsForLeaf;
import edu.uci.jforests.learning.trees.Tree;

/**
 * @author phong_nguyen
 *
 */
public class RegressionTreeLearner_MF extends RegressionTreeLearner {

	private int nb_factors;
	public boolean debug = false;
	/**
	 * 
	 */
	public RegressionTreeLearner_MF(int nb_factors) {
		super();
		this.nb_factors = nb_factors;
	}
	
	@Override
	protected CandidateSplitsForLeaf getNewCandidateSplitsForLeaf(int numFeatures, int numInstances) {
		return new RegressionCandidateSplitsForLeaf_MF(numFeatures, numInstances);
	}
	
	@Override
	protected Histogram getNewHistogram(Feature f) {
		return new RegressionHistogram(f, this.debug);
	}
	
	@Override
	protected Tree getNewTree() {
		RegressionTree_MF tree = new RegressionTree_MF();
		tree.init(maxLeaves, maxLeafOutput, nb_factors);
		return tree;
	}

}
