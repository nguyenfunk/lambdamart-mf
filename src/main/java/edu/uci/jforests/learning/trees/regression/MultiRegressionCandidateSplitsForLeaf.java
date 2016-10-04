/**
 * 
 */
package edu.uci.jforests.learning.trees.regression;

import org.ejml.simple.SimpleMatrix;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;

import edu.uci.jforests.dataset.RankingDataset;
import edu.uci.jforests.learning.trees.CandidateSplitsForLeaf;
import edu.uci.jforests.learning.trees.LeafInstances;
import edu.uci.jforests.learning.trees.TreeLeafInstances;
import edu.uci.jforests.sample.RankingMFSample;
import edu.uci.jforests.sample.Sample;
import edu.uci.jforests.util.Util;

/**
 * @author phong_nguyen
 *
 */
public class MultiRegressionCandidateSplitsForLeaf extends CandidateSplitsForLeaf {

	protected SimpleMatrix sumMultiTargets;
	protected SimpleMatrix multiTargets;
	
	//protected DoubleMatrix1D sumMultiTargets;
	//protected SparseDoubleMatrix2D multiTargets;
	
	int nb_factors;
	
	/**
	 * @param numFeatures
	 * @param numInstances
	 */
	public MultiRegressionCandidateSplitsForLeaf(int numFeatures,
			int numInstances, int nb_factors) {
		super(numFeatures, numInstances);
		
		bestSplitPerFeature = new MultiRegressionTreeSplit[numFeatures];
		for (int f = 0; f < numFeatures; f++) {
			bestSplitPerFeature[f] = new MultiRegressionTreeSplit();			
		}
		
		this.nb_factors = nb_factors;
		multiTargets = new SimpleMatrix(numInstances,nb_factors);
		//multiTargets = new SparseDoubleMatrix2D(numQueries,nb_factors);
		sumMultiTargets = new SimpleMatrix(1,nb_factors);
		
		//System.out.println("New CandidateLeafSplit "+ this+", "+numInstances);
		
	}
	
	//public DoubleMatrix1D getSumMultiTargets() {
	//	return sumMultiTargets.copy();
	public SimpleMatrix getSumMultiTargets() {
		return sumMultiTargets.copy();
	}

	@Override
	public void init(int curLeafIndex, TreeLeafInstances treeLeafInstances, Sample trainSet) {
		this.init(curLeafIndex);
		
		totalWeightedCount = 0;
		
		LeafInstances leafInstances = treeLeafInstances.getLeafInstances(curLeafIndex);
		numInstancesInLeaf = leafInstances.end - leafInstances.begin;
		
		//sumMultiTargets = new SimpleMatrix(1,nb_factors);
		//sumMultiTargets = new SparseDoubleMatrix1D(nb_factors);
		sumMultiTargets.zero();
		multiTargets.zero();
		
		RankingMFSample trainSample = (RankingMFSample)trainSet;
		
		for (int i = 0; i < numInstancesInLeaf; i++) {
			
			indices[i] = leafInstances.indices[leafInstances.begin + i];
			double weight = 1.0; //trainSet.weights[indices[i]];
			this.weights[i] = weight;
			
			/*
			int instanceIdx = trainSample.indicesInDataset[indices[i]];
			int idx = ((RankingDataset)trainSample.dataset).qdLines[instanceIdx];
			int didx = ((RankingDataset)trainSample.dataset).docIdx.get(idx); 
			*/
			SimpleMatrix target = trainSample.target.extractVector(true, indices[i]);
			
			multiTargets.insertIntoThis(i, 0, target);
			sumMultiTargets = sumMultiTargets.plus(target);
			//System.out.println(i+":"+sumMultiTargets);
			
			
			//DoubleMatrix1D target = ((RankingMFSample)trainSet).U.viewRow(indices[i]);
			//multiTargets.viewRow(indices[i]).assign(target);
			//multiTargets.viewRow(indices[i]).assign(target.getMatrix().getData());
			//sumMultiTargets.assign(multiTargets.viewRow(indices[i]), cern.jet.math.Functions.plus);
			
			totalWeightedCount += weight;
		}
		
		/*
		System.out.println();
		System.out.println("Init CandidateLeafSplit, "+ curLeafIndex+", "+ numInstancesInLeaf+"="+this);
		System.out.println(sumMultiTargets.elementSum());
		System.out.println(multiTargets.elementSum());
		Util.readline();
		*/
		//System.out.println(sumMultiTargets);
	}
	
	//public DoubleMatrix2D getMultiTargets() {
	//	return multiTargets.copy();
	public SimpleMatrix getMultiTargets() {
		return multiTargets;
	}
}
