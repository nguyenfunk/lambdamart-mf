/**
 * 
 */
package edu.uci.jforests.learning.trees.regression;

import org.ejml.simple.SimpleMatrix;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.jet.math.Functions;

import edu.uci.jforests.dataset.Feature;
import edu.uci.jforests.dataset.Histogram;
import edu.uci.jforests.learning.trees.CandidateSplitsForLeaf;
import edu.uci.jforests.util.Util;

/**
 * @author phong_nguyen
 *
 */
public class MultiRegressionHistogram extends Histogram {

	public int nb_factors;
	public boolean debug;
	public SimpleMatrix sumMultiTargets;
	public SimpleMatrix perValueSumMultiTargets;
	//public double[][] perValueSumMultiTargets; 
	
	//public DoubleMatrix1D sumMultiTargets;
	//public DoubleMatrix2D perValueSumMultiTargets;
	//static Functions F = cern.jet.math.Functions.functions;
	
	/**
	 * @param feature
	 */
	public MultiRegressionHistogram(Feature feature, int nb_factors, boolean debug) {
		super(feature);
		
		this.nb_factors = nb_factors;
		this.debug = debug;
		
		sumMultiTargets = new SimpleMatrix(1,nb_factors);
		perValueSumMultiTargets = new SimpleMatrix(numValues, nb_factors);
		//perValueSumMultiTargets = new double[numValues][nb_factors];
		
		//sumMultiTargets = new SparseDoubleMatrix1D(nb_factors); 
		//perValueSumMultiTargets = new SparseDoubleMatrix2D(numValues, nb_factors); 
	}
	
	@Override
	protected void initCustomData(CandidateSplitsForLeaf leafSplitCandidates,
			int[] instances) {
		MultiRegressionCandidateSplitsForLeaf rLeafSplitCandidates = (MultiRegressionCandidateSplitsForLeaf) leafSplitCandidates;
		sumMultiTargets = rLeafSplitCandidates.getSumMultiTargets();	
		
		//for(int i=0; i<numValues; i++) Arrays.fill(perValueSumMultiTargets[i], 0);
		
		//perValueSumMultiTargets.assign(0);
		perValueSumMultiTargets.zero();
		//System.out.println("Hist: "+this.feature.getName()+" "+instances.length+" "+totalCount);
		//Util.readline();
		//rLeafSplitCandidates.getMultiTargets().printDimensions();
		
		feature.bins.initHistogram(this, totalCount, 
				((MultiRegressionCandidateSplitsForLeaf)leafSplitCandidates).getMultiTargets(),
				leafSplitCandidates.getWeights(), leafSplitCandidates.getIndices(), instances);
		
		if(this.debug){
			System.out.println(feature.getName()+"="+totalCount+","+perValueSumMultiTargets.transpose());
			Util.readline();
		}
		
	}

	@Override
	protected void subtractCustomData(Histogram child) {
		
		MultiRegressionHistogram rChild = (MultiRegressionHistogram) child;
		sumMultiTargets = sumMultiTargets.minus(rChild.sumMultiTargets);
		//sumMultiTargets.assign(rChild.sumMultiTargets, F.minus);
		
		for (int i = 0; i < numValues; i++) {
			
			//perValueSumMultiTargets.viewRow(i).assign(rChild.perValueSumMultiTargets.viewRow(i), F.minus);
			
			SimpleMatrix v = perValueSumMultiTargets.extractVector(true, i);
			SimpleMatrix m = rChild.perValueSumMultiTargets.extractVector(true, i);
			perValueSumMultiTargets.insertIntoThis(i, 0, v.minus(m));	
			
			//for(int j=0; j<nb_factors; j++) perValueSumMultiTargets[i][j] -= rChild.perValueSumMultiTargets[i][j];
		}
	}

}
