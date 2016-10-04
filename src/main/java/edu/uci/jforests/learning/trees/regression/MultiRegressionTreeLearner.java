/**
 * 
 */
package edu.uci.jforests.learning.trees.regression;

import org.ejml.simple.SimpleMatrix;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;

import edu.uci.jforests.dataset.Feature;
import edu.uci.jforests.dataset.Histogram;
import edu.uci.jforests.learning.trees.CandidateSplitsForLeaf;
import edu.uci.jforests.learning.trees.Tree;
import edu.uci.jforests.learning.trees.TreeLearner;
import edu.uci.jforests.learning.trees.TreeSplit;
import edu.uci.jforests.util.Util;

/**
 * @author phong_nguyen
 *
 */
public class MultiRegressionTreeLearner extends TreeLearner {

	int nb_factors;
	public boolean debug = false;

	
	/**
	 * 
	 */
	public MultiRegressionTreeLearner(int nb_factors) {
		super("MultiRegressionTree");
		this.nb_factors = nb_factors;
	}
	
		@Override
	protected Tree getNewTree() {
		MultiRegressionTree tree = new MultiRegressionTree(nb_factors);
		tree.init(maxLeaves, this.nb_factors);
		return tree;
	}

	@Override
	protected TreeSplit getNewSplit() {
		return new MultiRegressionTreeSplit();
	}
	
	@Override
	protected CandidateSplitsForLeaf getNewCandidateSplitsForLeaf(int numFeatures, int numInstances) {
		MultiRegressionCandidateSplitsForLeaf m  = new MultiRegressionCandidateSplitsForLeaf(numFeatures, numInstances, this.nb_factors);
		return m;
	}
	
	@Override
	protected Histogram getNewHistogram(Feature f) {
		return new MultiRegressionHistogram(f, nb_factors, this.debug);
	}
	
	@Override
	protected void setBestThresholdForSplit(TreeSplit split, Histogram histogram) {
		
		MultiRegressionHistogram regHistogram = (MultiRegressionHistogram) histogram;
		
		SimpleMatrix bestSumLeftTargets = new SimpleMatrix(1, nb_factors);
		bestSumLeftTargets.set(Double.NaN);
		//DoubleMatrix1D bestSumLeftTargets = new SparseDoubleMatrix1D(nb_factors);
		//bestSumLeftTargets.assign(Double.NaN);

		double bestGain = Double.NEGATIVE_INFINITY;

		double bestWeightedLeftCount = -1;
		int bestThreshold = 0;

		SimpleMatrix sumLeftTargets = new SimpleMatrix(1, nb_factors);
		sumLeftTargets.set(0);
		//DoubleMatrix1D sumLeftTargets = new SparseDoubleMatrix1D(nb_factors);
		//sumLeftTargets.assign(0);
		
		int leftCount = 0;
		double weightedLeftCount = 0.0;
		int totalCount = histogram.totalCount;
		
		for (int t = 0; t < histogram.numValues - 1; t++) {
			leftCount += histogram.perValueCount[t];
			weightedLeftCount += histogram.perValueWeightedCount[t];
			sumLeftTargets = sumLeftTargets.plus(regHistogram.perValueSumMultiTargets.extractVector(true, t));
			
			/*
			double[] v = sumLeftTargets.getMatrix().getData();
			for(int i=0; i<nb_factors; i++){
				//v[i] += regHistogram.perValueSumMultiTargets[t][i];
				v[i] += regHistogram.perValueSumMultiTargets.get(t, i);
				sumLeftTargets.set(i, v[i]);
			}
			*/
			
			//sumLeftTargets.assign(regHistogram.perValueSumMultiTargets.viewRow(t), cern.jet.math.Functions.plus);
			
			if (leftCount < minInstancesPerLeaf || leftCount == 0) {
				continue;
			}
			int rightCount = totalCount - leftCount;
			
			if (rightCount < minInstancesPerLeaf || rightCount == 0) {
				break;
			}

			histogram.splittable = true;
			
			double weightedRightCount = histogram.totalWeightedCount - weightedLeftCount;
			SimpleMatrix sumRightTargets = regHistogram.sumMultiTargets.minus(sumLeftTargets);
			//DoubleMatrix1D sumRightTargets = regHistogram.sumMultiTargets.copy();
			//sumRightTargets.assign(sumLeftTargets, cern.jet.math.Functions.minus);
			
			//double currentGain = (sumLeftTargets.elementSum() * sumLeftTargets.elementSum()) / weightedLeftCount + (sumRightTargets.elementSum() * sumRightTargets.elementSum()) / weightedRightCount;
			double currentGain = Math.pow(sumLeftTargets.normF(), 2) / weightedLeftCount + Math.pow(sumRightTargets.normF(), 2) / weightedRightCount;
			//double currentGain = (sumLeftTargets.transpose().mult(sumLeftTargets).elementSum()) / weightedLeftCount + (sumRightTargets.transpose().mult(sumRightTargets).elementSum()) / weightedRightCount;
			
			
			//double currentGain = (sumLeftTargets.aggregate(cern.jet.math.Functions.plus, cern.jet.math.Functions.square) / weightedLeftCount)
			//+ (sumRightTargets.aggregate(cern.jet.math.Functions.plus, cern.jet.math.Functions.square) / weightedRightCount);
			
			//System.out.println(t+": "+sumLeftTargets.elementSum()+" "+weightedLeftCount+" "+currentGain);
			//Util.readline();
			
			if (currentGain > bestGain) {
				bestWeightedLeftCount = weightedLeftCount;
				bestSumLeftTargets = sumLeftTargets.copy();
				bestThreshold = t;
				bestGain = currentGain;
			}
		}
		
		//System.out.println();
		//System.out.println("Found: "+ split.feature+" "+ bestGain+" "+bestThreshold+" "+bestWeightedLeftCount+" "+ histogram.totalWeightedCount);
		//Util.readline();
		
		Feature feature = curTrainSet.dataset.features[split.feature];
		split.threshold = feature.upperBounds[bestThreshold];
		split.originalThreshold = feature.getOriginalValue(split.threshold);
		//split.leftCount = bestWeightedLeftCount;
		//split.rightCount = histogram.totalWeightedCount - bestWeightedLeftCount;
			
		MultiRegressionTreeSplit regressionSplit = (MultiRegressionTreeSplit) split;
		regressionSplit.leftMultiOutput = bestSumLeftTargets.divide(bestWeightedLeftCount);
		regressionSplit.rightMultiOutput = (regHistogram.sumMultiTargets.minus(bestSumLeftTargets)).divide(histogram.totalWeightedCount - bestWeightedLeftCount);
		
		/*
		regressionSplit.leftMultiOutput = bestSumLeftTargets.copy();
		regressionSplit.leftMultiOutput.assign(cern.jet.math.Functions.div(bestWeightedLeftCount));
		
		regressionSplit.rightMultiOutput = regHistogram.sumMultiTargets.copy();
		regressionSplit.rightMultiOutput.assign(bestSumLeftTargets, cern.jet.math.Functions.minus).assign(cern.jet.math.Functions.div(totalWeightedCount - bestWeightedLeftCount));
		*/
		
		if (bestSumLeftTargets.elementSum()==0){ 
			System.out.println("End: "+feature.getName()+", "+split.originalThreshold+","+bestWeightedLeftCount+","+regHistogram.perValueSumMultiTargets.elementSum());
			//System.out.println(regHistogram.perValueSumMultiTargets.transpose());
			//Util.readline();
			/*
			for(int t=0; t<=bestThreshold; t++)
				System.out.print(regHistogram.perValueSumMultiTargets.viewRow(t).get(0)+",");
			System.out.println();
			*/
		}
		
		//System.out.println(split.feature+", "+bestWeightedLeftCount+", "+"L:"+regressionSplit.leftMultiOutput.elementSum()+" ");
		//System.out.println(split.feature+", "+(histogram.totalWeightedCount - bestWeightedLeftCount)+", "+"R:"+regressionSplit.rightMultiOutput.elementSum());
		//System.out.println(split.feature+", "+histogram.totalWeightedCount+", "+" "+regHistogram.sumMultiTargets.elementSum());
		//Util.readline();
		
		//split.gain = bestGain - (regHistogram.sumMultiTargets.elementSum() * regHistogram.sumMultiTargets.elementSum()) / histogram.totalWeightedCount;
		split.gain = bestGain - Math.pow(regHistogram.sumMultiTargets.normF(), 2) / histogram.totalWeightedCount;
		//split.gain = bestGain - (regHistogram.sumMultiTargets.transpose().mult(regHistogram.sumMultiTargets).elementSum()) / histogram.totalWeightedCount;
		
		//split.gain = bestGain - regHistogram.sumMultiTargets.aggregate(cern.jet.math.Functions.plus, cern.jet.math.Functions.square) / totalWeightedCount;
	}
	
}
