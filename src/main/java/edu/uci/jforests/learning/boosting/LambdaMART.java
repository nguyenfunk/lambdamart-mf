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

package edu.uci.jforests.learning.boosting;

import java.util.Arrays;
import java.util.TreeSet;

import edu.uci.jforests.dataset.RankingDataset;
import edu.uci.jforests.eval.EvaluationMetric;
import edu.uci.jforests.eval.ranking.NDCGEval;
import edu.uci.jforests.learning.trees.LeafInstances;
import edu.uci.jforests.learning.trees.Tree;
import edu.uci.jforests.learning.trees.TreeLeafInstances;
import edu.uci.jforests.learning.trees.regression.RegressionTree;
import edu.uci.jforests.sample.RankingSample;
import edu.uci.jforests.sample.Sample;
import edu.uci.jforests.util.ArraysUtil;
import edu.uci.jforests.util.ConfigHolder;
import edu.uci.jforests.util.Constants;
import edu.uci.jforests.util.ScoreBasedComparator;
import edu.uci.jforests.util.concurrency.BlockingThreadPoolExecutor;
import edu.uci.jforests.util.concurrency.TaskCollection;
import edu.uci.jforests.util.concurrency.TaskItem;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

/**
 * @author Yasser Ganjisaffar <ganjisaffar at gmail dot com>
 */

public class LambdaMART extends GradientBoosting {

	private TaskCollection<LambdaWorker> workers;

	double[] maxDCG;
	double sigmoidParam;
	double[] sigmoidCache;
	double minScore;
	double maxScore;
	double sigmoidBinWidth;

	double[] denomWeights;

	int[] subLearnerSampleIndicesInTrainSet;
	
	protected TIntObjectHashMap<TIntDoubleHashMap> inputItemSims, outputItemSims;
	protected double lambdaInputItem, lambdaOutputItem;
	
	public LambdaMART(TIntObjectHashMap<TIntDoubleHashMap> inputItemSims, TIntObjectHashMap<TIntDoubleHashMap> outputItemSims, double lambdaInputItem, double lambdaOutputItem) {
		super("LambdaMART");
		this.inputItemSims = inputItemSims;
		this.outputItemSims = outputItemSims;
		this.lambdaInputItem = lambdaInputItem;
		this.lambdaOutputItem = lambdaOutputItem;
	}

	public LambdaMART() {
		super("LambdaMART");
	}

	public LambdaMART(String algorithmName) {
		super(algorithmName);
	}
	
	public void init(ConfigHolder configHolder, RankingDataset dataset, int maxNumTrainInstances, int maxNumValidInstances, EvaluationMetric evaluationMetric)
			throws Exception {
		super.init(configHolder, maxNumTrainInstances, maxNumValidInstances, evaluationMetric);

		LambdaMARTConfig lambdaMartConfig = configHolder.getConfig(LambdaMARTConfig.class);
		GradientBoostingConfig gradientBoostingConfig = configHolder.getConfig(GradientBoostingConfig.class);
		int[][] labelCountsPerQuery = NDCGEval.getLabelCountsForQueries(dataset.targets, dataset.queryBoundaries);
		maxDCG = NDCGEval.getMaxDCGForAllQueriesAtTruncation(dataset.targets, dataset.queryBoundaries, lambdaMartConfig.maxDCGTruncation, labelCountsPerQuery);

		// Sigmoid parameter is set to be equal to the learning rate.
		sigmoidParam = gradientBoostingConfig.learningRate;

		initSigmoidCache(lambdaMartConfig.sigmoidBins, lambdaMartConfig.costFunction);

		workers = new TaskCollection<LambdaWorker>();
		int numWorkers = BlockingThreadPoolExecutor.getInstance().getMaximumPoolSize();
		for (int i = 0; i < numWorkers; i++) {
			workers.addTask(new LambdaWorker(dataset.maxDocsPerQuery));
		}

		denomWeights = new double[maxNumTrainInstances];
		subLearnerSampleIndicesInTrainSet = new int[maxNumTrainInstances];	
	}

	protected void initSigmoidCache(int sigmoidBins, String costFunction) throws Exception {
		minScore = Constants.MIN_EXP_POWER / sigmoidParam;
		maxScore = -minScore;

		sigmoidCache = new double[sigmoidBins];
		sigmoidBinWidth = (maxScore - minScore) / sigmoidBins;
		if (costFunction.equals("cross-entropy")) {
			double score;
			for (int i = 0; i < sigmoidBins; i++) {
				score = minScore + i * sigmoidBinWidth;
				if (score > 0.0) {
					sigmoidCache[i] = 1.0 - 1.0 / (1.0 + Math.exp(-sigmoidParam * score));
				} else {
					sigmoidCache[i] = 1.0 / (1.0 + Math.exp(sigmoidParam * score));
				}
			}
		} else if (costFunction.equals("fidelity")) {
			double score;
			for (int i = 0; i < sigmoidBins; i++) {
				score = minScore + i * sigmoidBinWidth;
				if (score > 0.0) {
					double exp = Math.exp(-2 * sigmoidParam * score);
					sigmoidCache[i] = (-sigmoidParam / 2) * Math.sqrt(exp / Math.pow(1 + exp, 3));
				} else {
					double exp = Math.exp(sigmoidParam * score);
					sigmoidCache[i] = (-sigmoidParam / 2) * Math.sqrt(exp / Math.pow(1 + exp, 3));
				}
			}
		} else {
			throw new Exception("Unknown cost function: " + costFunction);
		}
	}

	@Override
	protected void preprocess() {
		Arrays.fill(trainPredictions, 0, curTrainSet.size, 0);
		if(curValidSet != null) Arrays.fill(validPredictions, 0, curValidSet.size, 0);
	}

	@Override
	protected void postProcessScores() {
		// Do nothing
	}

	protected double getAdjustedOutput(LeafInstances leafInstances) {
		double numerator = 0.0;
		double denomerator = 0.0;
		int instance;
		for (int i = leafInstances.begin; i < leafInstances.end; i++) {
			instance = subLearnerSampleIndicesInTrainSet[leafInstances.indices[i]];
			numerator += residuals[instance];
			denomerator += denomWeights[instance];
		}
		return (numerator + Constants.EPSILON) / (denomerator + Constants.EPSILON);
	}

	@Override
	protected void adjustOutputs(Tree tree, TreeLeafInstances treeLeafInstances) {
		
		LeafInstances leafInstances = new LeafInstances();
		for (int l = 0; l < tree.numLeaves; l++) {
			treeLeafInstances.loadLeafInstances(l, leafInstances);
			double adjustedOutput = getAdjustedOutput(leafInstances);
			((RegressionTree) tree).setLeafOutput(l, adjustedOutput);
		}
	}

	protected void setSubLearnerSampleWeights(RankingSample sample) {
		// Do nothing (weights are equal to 1 in LambdaMART)
	}

	@Override
	protected Sample getSubLearnerSample() {
		Arrays.fill(residuals, 0, curTrainSet.size, 0);
		Arrays.fill(denomWeights, 0, curTrainSet.size, 0);
		RankingSample trainSample = (RankingSample) curTrainSet;
		int chunkSize = 1 + (trainSample.numQueries / workers.getSize());
		int offset = 0;
		for (int i = 0; i < workers.getSize() && offset < trainSample.numQueries; i++) {
			int endOffset = offset + Math.min(trainSample.numQueries - offset, chunkSize);
			workers.getTask(i).init(offset, endOffset);
			BlockingThreadPoolExecutor.getInstance().execute(workers.getTask(i));
			offset += chunkSize;
		}
		BlockingThreadPoolExecutor.getInstance().await();

		trainSample = trainSample.getClone();
		trainSample.targets = residuals;
		
		setSubLearnerSampleWeights(trainSample);

		RankingSample zeroFilteredSample = trainSample.getClone();
		RankingSample subLearnerSample = zeroFilteredSample.getRandomSubSample(samplingRate, rnd);
		for (int i = 0; i < subLearnerSample.size; i++) {
			subLearnerSampleIndicesInTrainSet[i] = zeroFilteredSample.indicesInParentSample[subLearnerSample.indicesInParentSample[i]];
		}
		
		return subLearnerSample;
	}

	protected void onIterationEnd() {
		super.onIterationEnd();
	}

	protected class LambdaWorker extends TaskItem {

		int[] permutation;
		int[] labels;
		int beginIdx;
		int endIdx;
		ScoreBasedComparator comparator;

		public LambdaWorker(int maxDocsPerQuery) {
			permutation = new int[maxDocsPerQuery];
			labels = new int[maxDocsPerQuery];
			comparator = new ScoreBasedComparator();
		}

		public void init(int beginIdx, int endIdx) {
			this.beginIdx = beginIdx;
			this.endIdx = endIdx;
			comparator.labels = curTrainSet.targets;
		}

		//@Override
		public void run() {
			double scoreDiff;
			double rho;
			double deltaWeight;
			double pairWeight;
			double queryMaxDcg;
			RankingSample trainSet = (RankingSample) curTrainSet;
			double[] targets = trainSet.targets;
			comparator.scores = trainPredictions;
			
			try {
				for (int query = beginIdx; query < endIdx; query++) {
					int begin = trainSet.queryBoundaries[query];
					int numDocuments = trainSet.queryBoundaries[query + 1] - begin;
					queryMaxDcg = maxDCG[trainSet.queryIndices[query]];
					
					//TreeSet<Integer> ts = new TreeSet<Integer>();
					for (int i = 0; i < numDocuments; i++) {
						labels[i] = (int) targets[begin + i];
						/*
						if(!ts.contains(labels[i])){
							ts.add(labels[i]);
							if(ts.size()>((NDCGEval)evaluationMetric).evalTruncationLevel) 
								ts.remove(ts.first());
						}
						*/
					}
					
					comparator.offset = begin;
					for (int d = 0; d < numDocuments; d++) {
						permutation[d] = d;
					}					
					ArraysUtil.insertionSort(permutation, numDocuments, comparator);
					
					for (int i = 0; i < numDocuments; i++) {
						int betterIdx = permutation[i];
						if (labels[betterIdx] > 0) {
							
							
							int better_did = ((RankingDataset)trainSet.dataset).qdLines[begin+betterIdx];
							
							TIntDoubleHashMap inputItemNeighbours = null;
							if(inputItemSims != null){
								if(inputItemSims.containsKey(better_did)) inputItemNeighbours = inputItemSims.get(better_did);
							}
							TIntDoubleHashMap outputItemNeighbours = null;
							if(outputItemSims != null){
								if(outputItemSims.containsKey(better_did)) outputItemNeighbours = outputItemSims.get(better_did);
							}
							
							for (int j = 0; j < numDocuments; j++) {
								if (i != j) {
									int worseIdx = permutation[j];
									
									//if(!ts.contains(labels[betterIdx]) && !ts.contains(labels[worseIdx])) continue;
									
									if (labels[betterIdx] > labels[worseIdx]) {
										
										scoreDiff = trainPredictions[begin + betterIdx] - trainPredictions[begin + worseIdx];

										if (scoreDiff <= minScore) {
											rho = sigmoidCache[0];
										} else if (scoreDiff >= maxScore) {
											rho = sigmoidCache[sigmoidCache.length - 1];
										} else {
											rho = sigmoidCache[(int) ((scoreDiff - minScore) / sigmoidBinWidth)];
										}

										double w = 0;
										if (lambdaInputItem > 0 || lambdaOutputItem > 0){
											double w1=0, w2=0;
											int worse_did = ((RankingDataset)trainSet.dataset).qdLines[begin+worseIdx];
											if(inputItemNeighbours != null && inputItemNeighbours.containsKey(worse_did)) w1 = inputItemNeighbours.get(worse_did);
											if(outputItemNeighbours != null && outputItemNeighbours.containsKey(worse_did)) w2 = outputItemNeighbours.get(worse_did);
											w = (lambdaInputItem * w1 + lambdaOutputItem * w2) / (lambdaInputItem + lambdaOutputItem);
										}
										
										//System.out.println(better_prop_id+" "+worse_prop_id+" "+w);
										//Util.readline();
										
										pairWeight = (1-w) * (NDCGEval.GAINS[labels[betterIdx]] - NDCGEval.GAINS[labels[worseIdx]])
												* Math.abs((NDCGEval.discounts[i] - NDCGEval.discounts[j])) / queryMaxDcg;
										
										residuals[begin + betterIdx] += rho * pairWeight;
										residuals[begin + worseIdx] -= rho * pairWeight;

										deltaWeight = rho * (1.0 - rho) * pairWeight;
										denomWeights[begin + betterIdx] += deltaWeight;
										denomWeights[begin + worseIdx] += deltaWeight;
									}
								}
							}
						}
					}
				} // end for query
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
}
