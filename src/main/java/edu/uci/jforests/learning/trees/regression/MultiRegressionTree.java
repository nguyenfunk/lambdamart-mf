/**
 * 
 */
package edu.uci.jforests.learning.trees.regression;

import org.ejml.simple.SimpleMatrix;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;

import edu.uci.jforests.dataset.Dataset;
import edu.uci.jforests.learning.trees.Tree;
import edu.uci.jforests.learning.trees.TreeSplit;
import edu.uci.jforests.sample.Sample;
import edu.uci.jforests.util.Util;


/**
 * @author phong_nguyen
 *
 */
public class MultiRegressionTree extends Tree {

	private SimpleMatrix leafMultiOutputs;
	//private SparseDoubleMatrix2D leafMultiOutputs;
	int nb_factors;
	
	/**
	 * 
	 */
	public MultiRegressionTree(int nb_factors) {
		// TODO Auto-generated constructor stub
		this.nb_factors = nb_factors;
	}

	public void init(int maxLeaves, int k) {
		super.init(maxLeaves);
		leafMultiOutputs = new SimpleMatrix(maxLeaves, k);
		//leafMultiOutputs = new SparseDoubleMatrix2D(maxLeaves, k);
	}
	
	public SimpleMatrix getMultiOutput(Dataset dataset, int qidx) {
	//public double [] getOutput(Dataset dataset, int qidx) {
		int leaf = getLeaf(dataset, qidx);
		return leafMultiOutputs.extractVector(true, leaf); //.getMatrix().getData();
		//DoubleMatrix1D v = leafMultiOutputs.viewRow(leaf);
		//return v.toArray();
	}
	
	@Override
	public int split(int leaf, TreeSplit split) {
		int indexOfNewNonLeaf = super.split(leaf, split);
		MultiRegressionTreeSplit rsplit = (MultiRegressionTreeSplit) split;
		
		leafMultiOutputs.insertIntoThis(leaf, 0, rsplit.leftMultiOutput);
		leafMultiOutputs.insertIntoThis(numLeaves - 1, 0, rsplit.rightMultiOutput);
		
		//leafMultiOutputs.viewRow(leaf).assign(rsplit.leftMultiOutput);
		//leafMultiOutputs.viewRow(numLeaves - 1).assign(rsplit.rightMultiOutput);
		
		return indexOfNewNonLeaf;
	}

	@Override
	protected void addCustomData(String linePrefix, StringBuilder sb) {
		
		for (int n = 0; n < numLeaves; n++) {
			sb.append("\n" + linePrefix + "\t<LeafOutputs>"); 
			SimpleMatrix v = leafMultiOutputs.extractVector(true, n);
			//DoubleMatrix1D v = leafMultiOutputs.viewRow(n);
			for (int i=0; i<nb_factors; i++) sb.append(String.format("%.8f", v.get(i))+" ");
			sb.append("</LeafOutputs>");
		}
		
	}

	@Override
	public void loadCustomData(String str) throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void backfit(Sample sample) {
		// TODO Auto-generated method stub
		
	}
}
