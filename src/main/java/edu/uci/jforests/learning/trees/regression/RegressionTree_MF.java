/**
 * 
 */
package edu.uci.jforests.learning.trees.regression;

import org.ejml.simple.SimpleMatrix;

import edu.uci.jforests.dataset.Dataset;
import edu.uci.jforests.util.ArraysUtil;

/**
 * @author phong_nguyen
 *
 */
public class RegressionTree_MF extends RegressionTree {

	private SimpleMatrix leafMultiOutputs;
	public int nb_factors;
	
	/**
	 * 
	 */
	public RegressionTree_MF() {
		super();
	}
	
	public void init(int maxLeaves, double maxLeafOutput, int nb_factors) {
		super.init(maxLeaves, maxLeafOutput);
		this.leafMultiOutputs = new SimpleMatrix(maxLeaves, nb_factors);
		this.nb_factors = nb_factors;
	}
	
	public void setLeafMultiOutput(int leaf, SimpleMatrix output) {
		this.leafMultiOutputs.insertIntoThis(leaf, 0, output);
	}
	
	public SimpleMatrix getMultiOutput(Dataset dataset, int qidx) {
		int leaf = getLeaf(dataset, qidx);
		return leafMultiOutputs.extractVector(true, leaf); 
	}
	
	@Override
	protected void addCustomData(String linePrefix, StringBuilder sb) {
		
		for (int n = 0; n < numLeaves; n++) {
			sb.append("\n" + linePrefix + "\t<LeafOutputs>"); 
			SimpleMatrix v = leafMultiOutputs.extractVector(true, n);
			//DoubleMatrix1D v = leafMultiOutputs.viewRow(n);
			for (int i=0; i<v.numCols(); i++) sb.append(String.format("%.8f", v.get(i))+" ");
			sb.append("</LeafOutputs>");
		}
	}
	
	public void loadFromString(int numLeaves, String splitFeaturesLine, String leftChildrenLine, String rightChildrenLine, String thresholdsLine,
			String originalThresholdsLine, int nb_factors) throws Exception {
		splitFeatures = ArraysUtil.loadIntArrayFromLine(removeXmlTag(splitFeaturesLine, "SplitFeatures"), numLeaves - 1);
		leftChild = ArraysUtil.loadIntArrayFromLine(removeXmlTag(leftChildrenLine, "LeftChildren"), numLeaves - 1);
		rightChild = ArraysUtil.loadIntArrayFromLine(removeXmlTag(rightChildrenLine, "RightChildren"), numLeaves - 1);
		thresholds = ArraysUtil.loadIntArrayFromLine(removeXmlTag(thresholdsLine, "Thresholds"), numLeaves - 1);
		originalThresholds = ArraysUtil.loadDoubleArrayFromLine(removeXmlTag(originalThresholdsLine, "OriginalThresholds"), numLeaves - 1);
		this.numLeaves = numLeaves;
		this.leafMultiOutputs = new SimpleMatrix(numLeaves, nb_factors);
		this.nb_factors = nb_factors;
	}
	
	public void loadCustomData(String str, int index) throws Exception {
		double [] factors = ArraysUtil.loadDoubleArrayFromLine(removeXmlTag(str, "LeafOutputs"), nb_factors);
		this.leafMultiOutputs.setRow(index, 0, factors);
	}
}
