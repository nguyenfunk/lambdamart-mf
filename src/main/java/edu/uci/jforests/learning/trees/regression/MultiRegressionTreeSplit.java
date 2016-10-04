/**
 * 
 */
package edu.uci.jforests.learning.trees.regression;

import org.ejml.simple.SimpleMatrix;

import cern.colt.matrix.DoubleMatrix1D;

import edu.uci.jforests.learning.trees.TreeSplit;

/**
 * @author phong_nguyen
 *
 */
public class MultiRegressionTreeSplit extends TreeSplit {

	public SimpleMatrix leftMultiOutput;
    public SimpleMatrix rightMultiOutput;
    
    //public DoubleMatrix1D leftMultiOutput;
    //public DoubleMatrix1D rightMultiOutput;
    
    @Override
    public void copy(TreeSplit other) {
    	super.copy(other);
    	
    	this.leftMultiOutput = ((MultiRegressionTreeSplit) other).leftMultiOutput;
    	this.rightMultiOutput = ((MultiRegressionTreeSplit) other).rightMultiOutput;
    	
    	/*
    	if(((MultiRegressionTreeSplit) other).leftMultiOutput != null)
    		this.leftMultiOutput = ((MultiRegressionTreeSplit) other).leftMultiOutput.copy();
    	if(((MultiRegressionTreeSplit) other).rightMultiOutput != null)
    		this.rightMultiOutput = ((MultiRegressionTreeSplit) other).rightMultiOutput.copy();
    	*/
    }

}
