/**
 * 
 */
package edu.uci.jforests.learning.trees.regression;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import cern.jet.random.engine.RandomGenerator;

import edu.uci.jforests.learning.trees.LeafInstances;
import edu.uci.jforests.learning.trees.TreeLeafInstances;
import edu.uci.jforests.sample.RankingMFSample;
import edu.uci.jforests.sample.Sample;

/**
 * @author phong_nguyen
 *
 */
public class RegressionCandidateSplitsForLeaf_MF extends
		RegressionCandidateSplitsForLeaf {

	/**
	 * @param numFeatures
	 * @param numInstances
	 */
	public RegressionCandidateSplitsForLeaf_MF(int numFeatures, int numInstances) {
		super(numFeatures, numInstances);
	}
	
	@Override
	public void init(int curLeafIndex, TreeLeafInstances treeLeafInstances, Sample trainSet) {
		this.init(curLeafIndex);
		totalWeightedCount = 0;
		
		LeafInstances leafInstances = treeLeafInstances.getLeafInstances(curLeafIndex);
		numInstancesInLeaf = leafInstances.end - leafInstances.begin;
		
		RankingMFSample trainSample = (RankingMFSample)trainSet;
		//Random rnd = new Random();
		
		sumTargets = 0;
		for (int i = 0; i < numInstancesInLeaf; i++) {
			indices[i] = leafInstances.indices[leafInstances.begin + i];
			//double target = trainSet.targets[indices[i]];
			
			SimpleMatrix m_target = trainSample.target.extractVector(true, indices[i]);
			double target = m_target.elementSum(); 
			//double target = rnd.nextDouble();
			
			double weight = trainSet.weights[indices[i]];
			this.targets[i] = target;
			this.weights[i] = weight;
			sumTargets += target * weight;
			totalWeightedCount += weight;
		}
	}

}
