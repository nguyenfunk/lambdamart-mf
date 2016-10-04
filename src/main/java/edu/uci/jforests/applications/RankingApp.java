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

package edu.uci.jforests.applications;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import edu.uci.jforests.config.RankingTrainingConfig;
import edu.uci.jforests.dataset.Dataset;
import edu.uci.jforests.dataset.RankingDataset;
import edu.uci.jforests.dataset.RankingDatasetLoader;
import edu.uci.jforests.eval.EvaluationMetric;
import edu.uci.jforests.eval.ranking.NDCGEval;
import edu.uci.jforests.learning.LearningModule;
import edu.uci.jforests.learning.boosting.LambdaMART;
import edu.uci.jforests.sample.RankingSample;
import edu.uci.jforests.sample.Sample;
import edu.uci.jforests.util.Util;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

/**
 * @author Yasser Ganjisaffar <ganjisaffar at gmail dot com>
 */

public class RankingApp extends ClassificationApp {

	protected int maxDocsPerQuery;
	protected TIntObjectHashMap<TIntDoubleHashMap> inputItemSims, outputItemSims;
	protected double lambdaInputItem, lambdaOutputItem;

	public RankingApp() {
		super();
	}

	@Override
	protected void init() throws Exception {
		maxDocsPerQuery = ((RankingDataset) trainSet.dataset).maxDocsPerQuery;
		if (validSet != null) {
			maxDocsPerQuery = Math.max(maxDocsPerQuery, ((RankingDataset) validSet.dataset).maxDocsPerQuery);
		}
		NDCGEval.initialize(maxDocsPerQuery);
		super.init();

		String trainQidsFilename = ((RankingTrainingConfig) trainingConfig).trainQidsFilename;
		if (trainQidsFilename != null) {
			List<Integer> trainQids = Util.loadIntegersFromFile(trainQidsFilename);
			List<Integer> validQids = new ArrayList<Integer>();
			int validSize = (int) (trainQids.size() * 0.4);
			for (int i = 0; i < validSize; i++) {
				int idx = rnd.nextInt(trainQids.size());
				int qid = trainQids.get(idx);
				trainQids.remove(idx);
				validQids.add(qid);
			}
			Collections.sort(validQids);
			RankingSample newTrainSet = ((RankingSample) trainSet).getFilteredSubSample(trainQids);
			validSet = ((RankingSample) trainSet).getFilteredSubSample(validQids);
			trainSet = newTrainSet;
		}
		
		String inputItemSimilarityFiles = ((RankingTrainingConfig) trainingConfig).inputItemSimilarityFiles; 
		if(inputItemSimilarityFiles != null) this.inputItemSims = getSimilarities(inputItemSimilarityFiles.split(" "), ((RankingTrainingConfig) trainingConfig).inputItemNearestNeighbours);
		
		String outputItemSimilarityFiles = ((RankingTrainingConfig) trainingConfig).outputItemSimilarityFiles; 
		if(outputItemSimilarityFiles != null) this.outputItemSims = getSimilarities(outputItemSimilarityFiles.split(" "), ((RankingTrainingConfig) trainingConfig).outputItemNearestNeighbours);
		
		lambdaInputItem = ((RankingTrainingConfig) trainingConfig).lambdaInputItem;
		lambdaOutputItem = ((RankingTrainingConfig) trainingConfig).lambdaOutputItem;
	}

	@Override
	protected void loadConfig() {
		trainingConfig = new RankingTrainingConfig();
		trainingConfig.init(configHolder);
	}

	@Override
	protected Dataset newDataset() {
		return new RankingDataset();
	}

	@Override
	public void initDataset(Dataset dataset) throws Exception {
		if (dataset == null || !dataset.needsInitialization) {
			return;
		}
		RankingDataset rankingDataset = (RankingDataset) dataset;
		int[][] labelCounts = NDCGEval.getLabelCountsForQueries(rankingDataset.targets, rankingDataset.queryBoundaries);
		rankingDataset.maxDCG = NDCGEval.getMaxDCGForAllQueriesUptoTruncation(rankingDataset.targets, rankingDataset.queryBoundaries,
				NDCGEval.MAX_TRUNCATION_LEVEL, labelCounts);
	}

	@Override
	public void loadDataset(InputStream in, Dataset dataset) throws Exception {
		RankingDatasetLoader.load(in, (RankingDataset) dataset);
	}

	@Override
	protected LearningModule getLearningModule(String name) throws Exception {
		int maxTrainInstances = getMaxTrainInstances();		
		if (name.equals("LambdaMART")) {
			LambdaMART learner = new LambdaMART(inputItemSims, outputItemSims, lambdaInputItem, lambdaOutputItem);
			learner.init(configHolder, (RankingDataset) trainDataset, maxTrainInstances, (validDataset != null ? validDataset.numInstances
					: trainDataset.numInstances), evaluationMetric);
			return learner;
		} 
		else {
			return super.getLearningModule(name);
		}
	}

	@Override
	protected EvaluationMetric getEvaluationMetric(String name) throws Exception {
		if (name.equals("NDCG")) {
			return new NDCGEval(maxDocsPerQuery, ((RankingTrainingConfig) trainingConfig).validNDCGTruncation);
		}
		return super.getEvaluationMetric(name);
	}

	@Override
	protected Sample createSample(Dataset dataset, boolean trainSample) {
		RankingSample sample = new RankingSample((RankingDataset) dataset);
		RankingTrainingConfig config = (RankingTrainingConfig) trainingConfig;
		if (trainSample) {
			if (config.augmentationDocSamplingEnabled) {
				return sample.getAugmentedSampleWithDocSampling(config.augmentationDocSamplingTimes, config.augmentationDocSamplingRate, rnd);
			}
		}
		return sample;
	}

	@Override
	protected int getMaxTrainInstances() {
		RankingTrainingConfig config = (RankingTrainingConfig) trainingConfig;
		if (config.augmentationDocSamplingEnabled) {
			return trainDataset.numInstances * (config.augmentationDocSamplingTimes + 1);
		} else {
			return trainDataset.numInstances;
		}
	}

	protected TIntObjectHashMap<TIntDoubleHashMap> getSimilarities(String[] files, int nn) throws Exception{
		
		TIntObjectHashMap<TIntDoubleHashMap> sims = new TIntObjectHashMap<TIntDoubleHashMap>();
		
		System.out.println("Loading similarities.");
		
		for(String file: files){
			InputStream in = ioUtils.getInputStream(file);
			
			BufferedReader reader = new BufferedReader(new InputStreamReader(in));
			String line;
			while ((line = reader.readLine()) != null) {
				String [] tokens = line.split(" ");
				int key = Integer.parseInt(tokens[0].substring(0, tokens[0].length()-1));

				TIntDoubleHashMap neighbours = new TIntDoubleHashMap();
				int n = Math.min(nn+1, tokens.length);
				for(int i=1; i<n; i++){
					String[] data = tokens[i].split("=");
					neighbours.put( Integer.parseInt(data[0]) , Double.parseDouble(data[1]));
				}
			
				sims.put(key, neighbours);
			}
			reader.close();
			
			System.out.println("\t"+file);
		}
		
		System.out.println("done.");
		return sims;
	}
}
