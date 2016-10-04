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

package edu.uci.jforests.learning;

import org.ejml.simple.SimpleMatrix;

/*
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
*/

import edu.uci.jforests.dataset.RankingDataset;
import edu.uci.jforests.learning.trees.Ensemble;
import edu.uci.jforests.learning.trees.decision.DecisionTree;
import edu.uci.jforests.learning.trees.regression.EnsembleMF;
import edu.uci.jforests.learning.trees.regression.RegressionTree;
import edu.uci.jforests.learning.trees.regression.RegressionTree_MF;
import edu.uci.jforests.sample.RankingMFSample;
import edu.uci.jforests.sample.RankingSample;
import edu.uci.jforests.sample.Sample;
import edu.uci.jforests.util.Util;

/**
 * @author Yasser Ganjisaffar <ganjisaffar at gmail dot com>
 */

public class LearningUtils {
	
	public static void updateScores(Sample sampleSet, double[] scores, Ensemble ensemble) {
		updateScores(sampleSet, scores, ensemble, null);
	}
	
	public static void updateScores(Sample sampleSet, double[] scores, Ensemble ensemble, LearningProgressListener progressListener) {
		for (int t = 0; t < ensemble.getNumTrees(); t++) {
			RegressionTree tree = (RegressionTree) ensemble.getTreeAt(t);
			double treeWeight = ensemble.getWeightAt(t);
			for (int i = 0; i < sampleSet.size; i++) {
				scores[i] += treeWeight * tree.getOutput(sampleSet.dataset, sampleSet.indicesInDataset[i]);
			}
			if (progressListener != null) {
				progressListener.onScoreEval();
			}
		}
	}
	
	public static void updateScores(Sample sampleSet, double[] scores, RegressionTree tree, double treeWeight) {
		if (sampleSet.indicesInDataset == null) {
			for (int i = 0; i < sampleSet.size; i++) {
				scores[i] += treeWeight * tree.getOutput(sampleSet.dataset, i);
			}	
		} else {
			for (int i = 0; i < sampleSet.size; i++) {
				scores[i] += treeWeight * tree.getOutput(sampleSet.dataset, sampleSet.indicesInDataset[i]);
			}
		}
	}
	
	public static void updateScores(RankingMFSample sample, double[] scores, EnsembleMF ensemble, double learningRate){
		
		for (int t = 0; t < ensemble.getNumTrees(); t++) {
			/*
			RegressionTree_MF userTree = (RegressionTree_MF) ensemble.getTreeAt(t);
			for(int i=0; i<sample.numQueries; i++){
				SimpleMatrix u = sample.U.extractVector(true, i);
				SimpleMatrix g =  userTree.getMultiOutput(sample.dataset, sample.userIndicesInDataset[i]);
				sample.U.insertIntoThis(i, 0, u.plus(learningRate, g));
			}
			*/
			/*
			RegressionTree_MF itemTree = (RegressionTree_MF) ensemble.getItemTreeAt(t);
			for(int i=0; i<sample.numDocs; i++){
				SimpleMatrix v = sample.V.extractVector(true, i);
				SimpleMatrix g =  itemTree.getMultiOutput(sample.dataset, sample.itemIndicesInDataset[i]);
				sample.V.insertIntoThis(i, 0, v.plus(learningRate, g));
			}
			*/
		}
		
		for(int i=0; i<sample.numQueries; i++){
			
			int begin = sample.queryBoundaries[i];
			int numDocuments = sample.queryBoundaries[i + 1] - begin;
			
			SimpleMatrix u = sample.U.extractVector(true, i);
			
			for (int j=0; j<numDocuments; j++){
				
				// The item index among other items
				int idx = ((RankingDataset)sample.dataset).qdLines[begin+j];
				int didx = ((RankingDataset)sample.dataset).docIdx.get(idx); 
				SimpleMatrix v = sample.V.extractVector(true, didx);
				
				double r = u.dot(v.transpose());
				scores[begin+j] =  r; 		
			}
		}
	}
	
	public static SimpleMatrix updateScoresForU(int qidx, RankingMFSample sampleSet, RegressionTree_MF tree, double learningRate) {
	//public static SimpleMatrix updateScoresForU(int qidx, RankingMFSample sampleSet, MultiRegressionTree tree, double learningRate) {
	//public static void updateScoresForU(RankingMFSample sampleSet, MultiRegressionTree tree, double learningRate) {	
		
		//SimpleMatrix u = sampleSet.U.extractVector(true, qidx);
		int instanceIdx = ((RankingDataset)sampleSet.dataset).queryBoundaries[qidx];
		SimpleMatrix g = tree.getMultiOutput(sampleSet.dataset, instanceIdx);
		//u = u.plus(learningRate, g);
		//sampleSet.U.insertIntoThis(qidx, 0, u);
		
		return g; // u;
		
		/*
		//SimpleMatrix G = new SimpleMatrix(sampleSet.U.numRows(), sampleSet.U.numCols());
		//DoubleMatrix2D GU = new SparseDoubleMatrix2D(sampleSet.U.rows(), sampleSet.U.columns());
		
		for (int i = 0; i < sampleSet.numQueries; i++) {
			int instanceIdx = sampleSet.queryBoundaries[i];
			//double[] v = tree.getOutput(sampleSet.dataset, instanceIdx);
			//GU.viewRow(i).assign(v);
			//G.setRow(i, 0, v);
			
			SimpleMatrix vv = tree.getOutput(sampleSet.dataset, instanceIdx);
			//SimpleMatrix vv = new SimpleMatrix(1, sampleSet.U.numCols());
			//vv.setRow(0, 0, v);
			SimpleMatrix r = sampleSet.U.extractVector(true, i).plus(learningRate, vv);
			sampleSet.U.setRow(i, 0, r.getMatrix().getData());
		}
		
		//sampleSet.U = sampleSet.U.plus(learningRate, G);
		//sampleSet.U.assign(GU, cern.jet.math.Functions.plus);
		*/
	}
	
	public static SimpleMatrix updateScoresForV(int didx, RankingMFSample sampleSet, RegressionTree_MF tree, double learningRate) {
	//public static SimpleMatrix updateScoresForV(int didx, RankingMFSample sampleSet, MultiRegressionTree tree, double learningRate) {
	//public static void updateScoresForV(RankingMFSample sampleSet, MultiRegressionTree tree, double learningRate) {
		
		//SimpleMatrix v = sampleSet.V.extractVector(true, didx);
		int instanceIdx = ((RankingDataset)sampleSet.dataset).docBoundaries[didx];
		SimpleMatrix g = tree.getMultiOutput(sampleSet.dataset, instanceIdx);
		//v = v.plus(learningRate, g);
		//sampleSet.V.insertIntoThis(didx, 0, v);
		
		return g; // v;
		
		/*
		//SimpleMatrix G = new SimpleMatrix(sampleSet.V.numRows(), sampleSet.V.numCols());
		//DoubleMatrix2D GU = new SparseDoubleMatrix2D(sampleSet.U.rows(), sampleSet.U.columns());
		
		for (int i = 0; i < sampleSet.numDocs; i++) {
			int instanceIdx = ((RankingDataset)sampleSet.dataset).docBoundaries[i];
			//double[] v = tree.getOutput(sampleSet.dataset, instanceIdx);
			//GU.viewRow(i).assign(v);
			//G.setRow(i, 0, v);
			
			SimpleMatrix vv = tree.getOutput(sampleSet.dataset, instanceIdx);
			//SimpleMatrix vv = new SimpleMatrix(1, sampleSet.V.numCols());
			//vv.setRow(0, 0, v);
			
			SimpleMatrix r = sampleSet.V.extractVector(true, i).plus(learningRate, vv);
			sampleSet.V.setRow(i, 0, r.getMatrix().getData());
		}
		
		//sampleSet.V = sampleSet.V.plus(learningRate, G);
		//sampleSet.U.assign(GU, cern.jet.math.Functions.plus);
		*/ 
	}
	
	public static void updateDistributions(Sample sampleSet, double[][] dist, DecisionTree tree, double treeWeight) {
		for (int i = 0; i < sampleSet.size; i++) {
			double[] curDist = tree.getDistributionForInstance(sampleSet.dataset, sampleSet.indicesInDataset[i]);
			for (int c = 0; c < curDist.length; c++) {
				dist[i][c] += treeWeight * curDist[c];
			}
		}
	}

	public static void updateProbabilities(double[] prob, double[] scores, int size) {
		for (int i = 0; i < size; i++) {
			prob[i] = 1.0 / (1.0 + Math.exp(-2.0 * scores[i]));
		}
	}
	
	public static void updateProbabilities(double[] prob, double[] scores, int[] instances, int size) {
		for (int i = 0; i < size; i++) {
			int instance = instances[i];
			prob[instance] = 1.0 / (1.0 + Math.exp(-2.0 * scores[instance]));
		}
	}
	
}
