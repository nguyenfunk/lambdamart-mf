/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package edu.uci.jforests.dataset;

import org.ejml.simple.SimpleMatrix;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;

import edu.uci.jforests.dataset.NumericArrayFactory.NumericArrayType;
import edu.uci.jforests.learning.trees.decision.DecisionHistogram;
import edu.uci.jforests.learning.trees.regression.MultiRegressionHistogram;
import edu.uci.jforests.learning.trees.regression.RegressionHistogram;
import edu.uci.jforests.util.Util;

/**
 * @author Yasser Ganjisaffar <ganjisaffar at gmail dot com>
 */

public abstract class NumericArray implements ByteSerializable {

	protected int length;

	public NumericArray(int length) {
		this.length = length;
	}

	public int getLength() {
		return length;
	}

	public abstract NumericArrayType getType();

	public abstract int getBitsPerItem();

	//@Override
	public abstract int getSizeInBytes();

	//@Override
	public abstract int toByteArray(byte[] arr, int offset);

	//@Override
	public abstract int loadFromByteArray(byte[] arr, int offset);

	public abstract int get(int index);

	public abstract void set(int index, int value);

	public void initHistogram(RegressionHistogram histogram, int numInstancesInLeaf, double[] targets,
			double[] weights, int[] indices, int[] instances) {
		
		for (int i = 0; i < numInstancesInLeaf; i++) {
			int featureValue = get(instances[indices[i]]);
			histogram.perValueCount[featureValue]++;
			histogram.perValueWeightedCount[featureValue] += weights[i];
			histogram.perValueSumTargets[featureValue] += targets[i] * weights[i];
		}
		
	}
	
	public void initHistogram(DecisionHistogram histogram, int numInstancesInLeaf, double[] targets,
			double[] weights, int[] indices, int[] instances) {
		
		for (int i = 0; i < numInstancesInLeaf; i++) {
			int featureValue = get(instances[indices[i]]);
			histogram.perValueCount[featureValue]++;
			histogram.perValueWeightedCount[featureValue] += weights[i];
			histogram.perValueTargetDist[featureValue][(int)targets[i]] += weights[i];
		}
		
	}
	
	public void initHistogram(MultiRegressionHistogram histogram, int numInstancesInLeaf, SimpleMatrix targets,
	//public void initHistogram(MultiRegressionHistogram histogram, int numInstancesInLeaf, DoubleMatrix2D targets,
				double[] weights, int[] indices, int[] instances) {
		
		for (int i = 0; i < numInstancesInLeaf; i++) {
			
			//System.out.print(indices[i]+",");
			
			int featureValue = get(instances[indices[i]]);
			
			histogram.perValueCount[featureValue]++;
			histogram.perValueWeightedCount[featureValue] += weights[i];
			
			SimpleMatrix target = targets.extractVector(true, i);
			
			SimpleMatrix f = histogram.perValueSumMultiTargets.extractVector(true, featureValue).plus(weights[i], target);
			histogram.perValueSumMultiTargets.insertIntoThis(featureValue, 0, f);
		
			//for(int j=0; j<target.numCols(); j++) histogram.perValueSumMultiTargets[featureValue][j] += target.get(j); 
			
			//histogram.perValueSumMultiTargets.viewRow(featureValue).assign(targets.viewRow(i), cern.jet.math.Functions.plus);
			
			/*
			DoubleMatrix1D r = histogram.perValueSumMultiTargets.viewRow(featureValue);
			
			for(int j=0; j<histogram.nb_factors; j++){
				double v = r.get(j) + target.get(j);
				r.set(j, v);
			}
			*/
			
			/*
			histogram.perValueSumMultiTargets.viewRow(featureValue).assign(
					DoubleFactory1D.sparse.make(target.getMatrix().getData()), 
					cern.jet.math.Functions.plus);
			*/
		}
	}
	
	public abstract NumericArray getSubSampleNumericArray(int[] indices);
}
