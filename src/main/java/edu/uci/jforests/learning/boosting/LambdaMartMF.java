/**
 * 
 */
package edu.uci.jforests.learning.boosting;

import java.io.FileWriter;
import java.util.Arrays;
import java.util.TreeMap;
import java.util.TreeSet;

import org.ejml.simple.SimpleMatrix;

import edu.uci.jforests.dataset.RankingDataset;
import edu.uci.jforests.eval.EvaluationMetric;
import edu.uci.jforests.eval.ranking.NDCGEval;
import edu.uci.jforests.learning.LearningUtils;
import edu.uci.jforests.learning.trees.Ensemble;
import edu.uci.jforests.learning.trees.LeafInstances;
import edu.uci.jforests.learning.trees.Tree;
import edu.uci.jforests.learning.trees.TreeLeafInstances;
import edu.uci.jforests.learning.trees.TreesConfig;
import edu.uci.jforests.learning.trees.regression.EnsembleMF;
import edu.uci.jforests.learning.trees.regression.MultiRegressionTree;
import edu.uci.jforests.learning.trees.regression.MultiRegressionTreeLearner;
import edu.uci.jforests.learning.trees.regression.RegressionTreeLearner_MF;
import edu.uci.jforests.learning.trees.regression.RegressionTree_MF;
import edu.uci.jforests.sample.RankingMFSample;
import edu.uci.jforests.sample.Sample;
import edu.uci.jforests.util.ArraysUtil;
import edu.uci.jforests.util.ConfigHolder;
import edu.uci.jforests.util.Util;
import edu.uci.jforests.util.concurrency.BlockingThreadPoolExecutor;
import edu.uci.jforests.util.concurrency.TaskCollection;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

/**
 * @author phong_nguyen
 *
 */
public class LambdaMartMF extends LambdaMART {

	private TaskCollection<LambdaMFWorker> workers;
	private SimpleMatrix GU, GV;
	//private DoubleMatrix2D GU, GV;
	public int nb_factors;
	private double mu1, userSamplingRate, itemSamplingRate; 
	private ConfigHolder configHolder;
	private TreeSet<Integer> qids;
	private boolean debug;

	protected TIntObjectHashMap<TIntDoubleHashMap> inputUserSims, outputUserSims;
	protected double lambdaInputUser, lambdaOutputUser;
	private RankingMFSample subLearnerSample;
	
	public LambdaMartMF(TIntObjectHashMap<TIntDoubleHashMap> inputItemSims, TIntObjectHashMap<TIntDoubleHashMap> outputItemSims, double lambdaInputItem, double lambdaOutputItem,
			TIntObjectHashMap<TIntDoubleHashMap> inputUserSims, TIntObjectHashMap<TIntDoubleHashMap> outputUserSims, double lambdaInputUser, double lambdaOutputUser) {
		super(inputItemSims, outputItemSims, lambdaInputItem, lambdaOutputItem);
		this.inputUserSims = inputUserSims;
		this.outputUserSims = outputUserSims;
		this.lambdaInputUser = lambdaInputUser;
		this.lambdaOutputUser = lambdaOutputUser;
	}

	public LambdaMartMF() {
		super("LambdaMartMF");
	}
	
	@Override
	public void init(ConfigHolder configHolder, RankingDataset dataset, int maxNumTrainInstances, int maxNumValidInstances, EvaluationMetric evaluationMetric)
	throws Exception {
		
		super.init(configHolder, dataset, maxNumTrainInstances, maxNumValidInstances, evaluationMetric);
		
		LambdaMartMFConfig lambdaMartMFConfig = configHolder.getConfig(LambdaMartMFConfig.class);
		nb_factors = lambdaMartMFConfig.nb_factors;
		mu1 = lambdaMartMFConfig.mu1;
		userSamplingRate = lambdaMartMFConfig.userSamplingRate;
		itemSamplingRate = lambdaMartMFConfig.itemSamplingRate;
		debug = lambdaMartMFConfig.debug;
		
		workers = new TaskCollection<LambdaMFWorker>();
		int numWorkers = BlockingThreadPoolExecutor.getInstance().getMaximumPoolSize();
		for (int i = 0; i < numWorkers; i++) {
			workers.addTask(new LambdaMFWorker(dataset.maxDocsPerQuery));
		}
		
		this.configHolder = configHolder;
		
		qids = new TreeSet<Integer>();
		for(int qid : dataset.queryIdx){ 
			qids.add(qid);
		}
	}
	
	@Override
	public void postProcess(Tree tree, TreeLeafInstances treeLeafInstances) {
		
		LeafInstances leafInstances = new LeafInstances();
		SimpleMatrix output = new SimpleMatrix(1, nb_factors);
		
		for (int l = 0; l < tree.numLeaves; l++) {
			treeLeafInstances.loadLeafInstances(l, leafInstances);
			output.zero();
			
			for (int i = leafInstances.begin; i < leafInstances.end; i++) {
				int instance = leafInstances.indices[i];
				output = output.plus(subLearnerSample.target.extractVector(true, instance));
			}
			
			output = output.divide(leafInstances.end-leafInstances.begin);
			((RegressionTree_MF) tree).setLeafMultiOutput(l, output);
		}
		
	}
	
	@Override
	public Ensemble learn(Sample trainSet, Sample validSet) throws Exception {
		
		curTrainSet = trainSet;
		curValidSet = validSet;
		
		preprocess();
		
		RankingMFSample trainSample = (RankingMFSample)trainSet;
		trainSample.initFactors(nb_factors, rnd);
		
		RankingMFSample validSample = null;
		TreeSet<Integer> intersection = null;
		if(validSet != null){ 
			validSample = (RankingMFSample)validSet;
			validSample.initFactors(nb_factors, rnd);
			
			// Copy the V factors for the valid set which are in the train set 
			intersection = new TreeSet<Integer>(((RankingDataset)trainSample.dataset).docIdx.keySet());
			intersection.retainAll(((RankingDataset)validSample.dataset).docIdx.keySet());
			System.out.println("V intersection size = "+intersection.size());
			for(int i : intersection){
				int tidx = ((RankingDataset)trainSample.dataset).docIdx.get(i);
				int vidx = ((RankingDataset)validSample.dataset).docIdx.get(i);
				SimpleMatrix v = trainSample.V.extractVector(true, tidx);
				validSample.V.insertIntoThis(vidx, 0, v);
			}
		}
		
		GU = new SimpleMatrix( trainSample.numQueries, nb_factors);
		GV = new SimpleMatrix( trainSample.numDocs, nb_factors);
		//GU = new SparseDoubleMatrix2D( trainSample.numQueries, nb_factors);
		//GV = new SparseDoubleMatrix2D( numDocs, nb_factors);
		
		LambdaMartMFConfig lambdaMartMFConfig = configHolder.getConfig(LambdaMartMFConfig.class);
		TreesConfig treesConfig = configHolder.getConfig(TreesConfig.class);
		
		int subSampleNumQueries = (int) (trainSample.numQueries * userSamplingRate);
		int subSampleNumDocs = (int) (trainSample.numDocs * itemSamplingRate);
		
		//MultiRegressionTreeLearner userLearner = new MultiRegressionTreeLearner( nb_factors);
		RegressionTreeLearner_MF userLearner = new RegressionTreeLearner_MF(nb_factors);
		treesConfig.featuresToInclude = lambdaMartMFConfig.userFeatures;
		userLearner.init(trainSample.dataset, configHolder, subSampleNumQueries);
		userLearner.setParentModule(this);
		
		//MultiRegressionTreeLearner itemLearner = new MultiRegressionTreeLearner( nb_factors);
		RegressionTreeLearner_MF itemLearner = new RegressionTreeLearner_MF(nb_factors);
		treesConfig.featuresToInclude = lambdaMartMFConfig.itemFeatures;
		itemLearner.init(trainSample.dataset, configHolder, subSampleNumDocs );
		itemLearner.setParentModule(this);
		
		//itemLearner.debug = true;
		
		EnsembleMF ensemble = new EnsembleMF();
		//Ensemble ensemble = new Ensemble();
		bestValidationMeasurement = Double.NaN;
		int earlyStoppingIteration = 0;
		int bestIteration = 0;
		int[] treeCounts = new int[numSubModules];
		
		SimpleMatrix bestU = trainSample.U.copy();
		SimpleMatrix bestV = trainSample.V.copy();
		
		for (curIteration = 1; curIteration <= numSubModules; curIteration++) {
			
			subLearnerSample = getSubLearnerSample();
			
			subLearnerSample.indicesInDataset = subLearnerSample.itemIndicesInDataset;
			subLearnerSample.target = GV;
			Ensemble ie = itemLearner.learn(subLearnerSample, validSet);
			
			if (debug) System.out.println(ie);
			
			subLearnerSample.indicesInDataset = subLearnerSample.userIndicesInDataset;
			subLearnerSample.target = GU;
			Ensemble ue = userLearner.learn(subLearnerSample, validSet);
			
			if(debug) System.out.println(ue);
			//Util.readline();
			
			if (ue == null || ie == null) {
				break;
			}
			
			//MultiRegressionTree userTree = (MultiRegressionTree)ue.getTreeAt(0);
			//MultiRegressionTree itemTree = (MultiRegressionTree)ie.getTreeAt(0);
			RegressionTree_MF userTree = (RegressionTree_MF)ue.getTreeAt(0);
			RegressionTree_MF itemTree = (RegressionTree_MF)ie.getTreeAt(0);
			
			// For non cold start, we do not save the trees
			ensemble.addTree(userTree, ue.getWeightAt(0));
			ensemble.addItemTree(itemTree, ie.getWeightAt(0));
			treeCounts[curIteration - 1] = ensemble.getNumTrees();
			
			//trainSample.U = trainSample.U.plus(learningRate, GU);
			//trainSample.V = trainSample.V.plus(learningRate, GV);
			//updateScores(trainSample, null, null, trainPredictions, learningRate);
			updateScores(trainSample, userTree, itemTree, trainPredictions, learningRate);
			
			if (validSet == null) {
				earlyStoppingIteration = curIteration;
				printTrainMeasurement(curIteration, getTrainMeasurement());
			} else {
				
				validSample.U = trainSample.U.copy();
				for(int i : intersection){
					int tidx = ((RankingDataset)trainSample.dataset).docIdx.get(i);
					int vidx = ((RankingDataset)validSample.dataset).docIdx.get(i);
					SimpleMatrix v = trainSample.V.extractVector(true, tidx);
					validSample.V.insertIntoThis(vidx, 0, v);
				}
				
				//updateScores(validSample, null, null, validPredictions, learningRate);
				updateScores(validSample, userTree, itemTree, validPredictions, learningRate);
				
				double validMeasurement = getValidMeasurement();
				if (evaluationMetric.isFirstBetter(validMeasurement, bestValidationMeasurement, earlyStoppingTolerance)) {
					earlyStoppingIteration = curIteration;
					if (evaluationMetric.isFirstBetter(validMeasurement, bestValidationMeasurement, 0)) {
						bestValidationMeasurement = validMeasurement;
						bestIteration = curIteration;
						
						// For non cold start setting, we keep track of the best model to use it for test
						bestU = trainSample.U.copy();
						bestV = trainSample.V.copy();
					}
				}

				if (curIteration - bestIteration > this.earlyStoppingStep) {
					break;
				}
				
				if (printIntermediateValidMeasurements) {
					printTrainAndValidMeasurement(curIteration, validMeasurement, getTrainMeasurement(), evaluationMetric);
				}
			}
			
			onIterationEnd();
			
			if (debug) Util.readline();
		}
		
		if (earlyStoppingIteration > 0) {
			int treesToKeep = treeCounts[earlyStoppingIteration - 1];
			int treesToDelete = ensemble.getNumTrees() - treesToKeep;
			ensemble.removeLastTrees(treesToDelete);
			ensemble.removeLastItemTrees(treesToDelete);
		}
		
		// Save matrix models, to be cleaned
		if(bestU != null && bestV != null){
			bestU.saveToFileCSV("U.train.csv");
			bestV.saveToFileCSV("V.train.csv");
		
			FileWriter file = new FileWriter("train.docIdx.txt");
			
			String docIds = ((RankingDataset)trainSample.dataset).docIdx.keySet().toString();
			docIds = docIds.substring(1,docIds.length()-1);
			file.write(docIds+"\n");
			file.close();
		}
		
		onLearningEnd();
		return ensemble;
	}
	
	@Override
	protected RankingMFSample getSubLearnerSample() {
		
		GU.zero(); GV.zero();
		//GU.assign(0); GV.assign(0);
		RankingMFSample trainSample = (RankingMFSample)curTrainSet;
		
		int chunkSize = 1 + (trainSample.numQueries / workers.getSize());
		int offset = 0;
		for (int i = 0; i < workers.getSize() && offset < trainSample.numQueries; i++) {
			int endOffset = offset + Math.min(trainSample.numQueries - offset, chunkSize);
			workers.getTask(i).init(offset, endOffset);
			BlockingThreadPoolExecutor.getInstance().execute(workers.getTask(i));
			offset += chunkSize;
		}
		BlockingThreadPoolExecutor.getInstance().await();

		TreeMap<Integer,Integer> docIdx = ((RankingDataset)trainSample.dataset).docIdx;
	
		if(lambdaInputItem > 0 || lambdaOutputItem > 0)
		for(int didx : docIdx.keySet()){
			
			int i = docIdx.get(didx);
			SimpleMatrix v = trainSample.V.extractVector(true, i);
			SimpleMatrix gv = GV.extractVector(true, i);
			
			if(lambdaInputItem > 0 && inputItemSims != null){
				TIntDoubleHashMap itemNeighbours = inputItemSims.get(didx);
				
				if(itemNeighbours != null){
					//System.out.print(didx+": ");
					for(int nqid : itemNeighbours.keySet().toArray()){
						if(docIdx.containsKey(nqid)){
							double w = itemNeighbours.get(nqid);
							int nquery = docIdx.get(nqid); // get the index of the nearest query
							SimpleMatrix nv = trainSample.V.extractVector(true, nquery);
							gv = gv.plus(-lambdaInputItem, v.minus(nv).scale(w));
							
							//System.out.print(nqid+"="+w+",");
						}
						//else System.out.print(nqid+" not found,");
					}
					//System.out.println();
					//Util.readline();
				}
			}
			
			if(lambdaOutputItem > 0 && outputItemSims != null){
				TIntDoubleHashMap itemNeighbours = outputItemSims.get(didx);
				
				if(itemNeighbours != null){
					//System.out.print(didx+": ");
					for(int nqid : itemNeighbours.keySet().toArray()){
						if(docIdx.containsKey(nqid)){
							double w = itemNeighbours.get(nqid);
							int nquery = docIdx.get(nqid); // get the index of the nearest query
							SimpleMatrix nv = trainSample.V.extractVector(true, nquery);
							gv = gv.plus(-lambdaOutputItem, v.minus(nv).scale(w));
							
							//System.out.print(nqid+"="+w+",");
						}
						//else System.out.print(nqid+" not found,");
					}
					//System.out.println();
					//Util.readline();
				}
			}
		}
		
		GU = GU.plus(-mu1, trainSample.U);
		GV = GV.plus(-mu1, trainSample.V);
		
		//RankingMFSample subLearnerSample = trainSample.getRandomUserSubSample(userSamplingRate, rnd).getRandomItemSubSample(itemSamplingRate, rnd);
		
		//return subLearnerSample;
		return trainSample;
	}
	
	@Override
	public void printTrainMeasurement(int iteration, double trainMeasurement){
		RankingMFSample trainSample = (RankingMFSample)curTrainSet;
		//double lr = 0;
		//if (kernelUser != null) lr = trainSample.U.transpose().mult(kernelUser).mult(trainSample.U).trace();
		double nu = trainSample.U.normF(); // *(mu1/2)
		double nv = trainSample.V.normF() ; // (mu1/2)*
		double all = (1-trainMeasurement) + nu + nv;
		
		System.out.println(iteration 
				+ "\t" + "All:"+all
				+ "\t"  + "NDCG:"+trainMeasurement 
				+ "\t" + "U:"+nu 
				+ "\t" + "V:"+nv
				//+ "\t" + (mu1/2)*trainSample.U.aggregate(cern.jet.math.Functions.plus, cern.jet.math.Functions.square) 
				//+ "\t" + (mu1/2)*trainSample.V.aggregate(cern.jet.math.Functions.plus, cern.jet.math.Functions.square) 
				//+ "\t" + (mu2/2)*lr
		);
	}
	
	@Override
	public void printTrainAndValidMeasurement(int iteration, double validMeasurement, double trainMeasurement, EvaluationMetric evaluationMetric) {
		if (evaluationMetric.isFirstBetter(validMeasurement, bestPrintedValidMeasurement, 0)) {
			bestPrintedValidMeasurement = validMeasurement;
		}
		
		if (Double.isNaN(trainMeasurement)) {
			System.out.println(algorithmName + ": [Iteration: " + iteration + ", Valid: " + validMeasurement + ", Best: " + bestPrintedValidMeasurement
					+ "]");
		} else {
			RankingMFSample trainSample = (RankingMFSample)curTrainSet;
			RankingMFSample validSample = (RankingMFSample)curValidSet;
			
			double tnu = trainSample.U.normF();
			double tnv = trainSample.V.normF();
			double vnu = validSample.U.normF();
			double vnv = validSample.V.normF();
			
			System.out.println(iteration 
					+ "\t"  + "NDCG:"+trainMeasurement+" \t"+validMeasurement 
					+ "\t" + "U:"+ tnu +"\t"+ vnu 
					+ "\t" + "V:"+ tnv +"\t"+ vnv
			);
			
			if(Double.isNaN(vnu)){
				//System.out.println(validSample.U);
				//Util.readline();
				System.exit(-1);
			}
			
			if(Double.isNaN(vnv)){
				//System.out.println(validSample.V);
				//Util.readline();
				System.exit(-1);
			}
		}
	}
	
	public void updateScores(RankingMFSample sample, RegressionTree_MF userTree, RegressionTree_MF itemTree, double[] scores, double learningRate){
	//public void updateScores(RankingMFSample sample, MultiRegressionTree userTree, MultiRegressionTree itemTree, double[] scores, double learningRate){
		
		boolean[] docUpdates = new boolean[sample.numDocs];
		Arrays.fill(docUpdates, false);
		
		for(int i=0; i<sample.numQueries; i++){
			
			int begin = sample.queryBoundaries[i];
			int numDocuments = sample.queryBoundaries[i + 1] - begin;
			
			SimpleMatrix u = sample.U.extractVector(true, i);
			//SimpleMatrix g = LearningUtils.updateScoresForU(i, sample, userTree, learningRate);
			
			SimpleMatrix g =  userTree.getMultiOutput(sample.dataset, begin);
			sample.U.insertIntoThis(i, 0, u.plus(learningRate, g));
			
			//int qidx = sample.queryIdx[i];
			
			for (int j=0; j<numDocuments; j++){
				
				// The item index among other items
				int idx = ((RankingDataset)sample.dataset).qdLines[begin+j];
				int didx = ((RankingDataset)sample.dataset).docIdx.get(idx); 
				
				SimpleMatrix v = null; 
				if(!docUpdates[didx]){
					//v = LearningUtils.updateScoresForV(didx, sample, itemTree, learningRate);
					
					v = sample.V.extractVector(true, didx);
					//g = LearningUtils.updateScoresForV(didx, sample, itemTree, learningRate);
					int instanceIdx = ((RankingDataset)sample.dataset).docBoundaries[didx];
					g = itemTree.getMultiOutput(sample.dataset, instanceIdx);
					sample.V.insertIntoThis(didx, 0, v.plus(learningRate, g));
					docUpdates[didx] = true;
				} else 
				v = sample.V.extractVector(true, didx);
				
				double r = u.dot(v.transpose());
				
				/*
				if(debug) {
					int instanceIdx = ((RankingDataset)sample.dataset).docBoundaries[didx];
					System.out.print(qidx+","+idx+","+didx+","+instanceIdx+"=");
					System.out.println(r+","+(begin+j));
					Util.readline();
				}
				*/
				
				//SimpleMatrix r = sample.U.extractVector(true, i).mult(sample.V.extractVector(true, idx).transpose());
				
				scores[begin+j] =  r; 
			}
		}
		
	}
	
	private class LambdaMFWorker extends LambdaWorker {

		public LambdaMFWorker(int maxDocsPerQuery) {
			super(maxDocsPerQuery);
		}
		
		@Override
		public void run() {
			
			double scoreDiff;
			double rho;
			double pairWeight;
			double queryMaxDcg;
			RankingMFSample trainSet = (RankingMFSample) curTrainSet;
			double[] targets = trainSet.targets;
			comparator.scores = trainPredictions;
			
			try {
				
				for (int query = beginIdx; query < endIdx; query++) {
					int begin = trainSet.queryBoundaries[query];
					//int idx = trainSet.queryIdx[query];
					
					int numDocuments = trainSet.queryBoundaries[query + 1] - begin;
					queryMaxDcg = maxDCG[trainSet.queryIndices[query]];
					
					SimpleMatrix u = trainSet.U.extractVector(true, query);
					SimpleMatrix gu = GU.extractVector(true, query);
					
					if(inputUserSims != null){
						int qid = ((RankingDataset)trainSet.dataset).queryIdx[query];
						TIntDoubleHashMap userNeighbours = inputUserSims.get(qid);
						
						if(userNeighbours != null){
							//System.out.print(qid+": ");
							for(int nqid : userNeighbours.keySet().toArray()){
								if(qids.contains(nqid)){
									double w = userNeighbours.get(nqid);
									int nquery = qids.headSet(nqid).size(); // get the index of the nearest query
									SimpleMatrix nu = trainSet.U.extractVector(true, nquery);
									gu = gu.plus(-lambdaInputUser, u.minus(nu).scale(w));
									
									//System.out.print(nqid+"="+nquery+",");
								}
								//else System.out.print(nqid+" not found,");
							}
							//System.out.println();
							//Util.readline();
						}
					}
					
					if(outputUserSims != null){
						int qid = ((RankingDataset)trainSet.dataset).queryIdx[query];
						TIntDoubleHashMap userNeighbours = outputUserSims.get(qid);
						
						if(userNeighbours != null){
							//System.out.print(qid+": ");
							for(int nqid : userNeighbours.keySet().toArray()){
								if(qids.contains(nqid)){
									double w = userNeighbours.get(nqid);
									int nquery = qids.headSet(nqid).size(); // get the index of the nearest query
									SimpleMatrix nu = trainSet.U.extractVector(true, nquery);
									gu = gu.plus(-lambdaOutputUser, u.minus(nu).scale(w));
									
									//System.out.print(nqid+"="+nquery+",");
								}
								//else System.out.print(nqid+" not found,");
							}
							//System.out.println();
							//Util.readline();
						}
					}
					
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
					
					//System.out.println(ts);
					
					comparator.offset = begin;
					for (int d = 0; d < numDocuments; d++) {
						permutation[d] = d;
					}					
					
					ArraysUtil.insertionSort(permutation, numDocuments, comparator);
					
					for (int i = 0; i < numDocuments; i++) {
						int betterIdx = permutation[i];
						
						if (labels[betterIdx] == 0) continue;
						
						int dil = ((RankingDataset)trainSet.dataset).qdLines[begin+betterIdx];
						int did = ((RankingDataset)trainSet.dataset).docIdx.get(dil);
						
						SimpleMatrix gvi = GV.extractVector(true, did);
						SimpleMatrix vi = trainSet.V.extractVector(true, did);
						
						int better_did = ((RankingDataset)trainSet.dataset).qdLines[begin+betterIdx];
						
						TIntDoubleHashMap inputItemNeighbours = null;
						if(inputItemSims != null){
							if(inputItemSims.containsKey(better_did)){ 
								inputItemNeighbours = inputItemSims.get(better_did);
							
								/*
								// Semi-supervised regularizer: w_ij ( <u, v_i> - <u, v_j> )^2
								// where v_j is in the neighborhood of v_i and w_ij is the similarity between the two items
								// The SS regularizer forces similar item pairs to have similar output
								for(int nid : inputItemNeighbours.keys()){
								
									double w = inputItemNeighbours.get(nid);
									int dnd = ((RankingDataset)trainSet.dataset).docIdx.get(nid);
									SimpleMatrix vn = trainSet.V.extractVector(true, dnd);
									SimpleMatrix diff = vi.minus(vn);
									//gu = gu.plus(-lambdaInputItem, diff.scale(w * u.dot( diff.transpose() )) );
								}
								*/
							}
						}
						
						TIntDoubleHashMap outputItemNeighbours = null;
						if(outputItemSims != null){
							if(outputItemSims.containsKey(better_did)){ 
								outputItemNeighbours = outputItemSims.get(better_did);
							
								/*
								for(int nid : outputItemNeighbours.keys()){
								
									double w = outputItemNeighbours.get(nid);
									int dnd = ((RankingDataset)trainSet.dataset).docIdx.get(nid);
									SimpleMatrix vn = trainSet.V.extractVector(true, dnd);
									SimpleMatrix diff = vi.minus(vn);
									//gu = gu.plus(-lambdaOutputItem, diff.scale(w * u.dot( diff.transpose() )) );
								}
								*/
							}
						}
							
						for (int j = 0; j < numDocuments; j++) {
								
							if (i == j) continue; 
								
							int worseIdx = permutation[j];
							
							int djl = ((RankingDataset)trainSet.dataset).qdLines[begin+worseIdx];
							int djd = ((RankingDataset)trainSet.dataset).docIdx.get(djl);
							
							SimpleMatrix gvj = GV.extractVector(true, djd);
							SimpleMatrix vj = trainSet.V.extractVector(true, djd);
							
							//if(!ts.contains(labels[betterIdx]) && !ts.contains(labels[worseIdx])) continue;
								
							double w = 0;
							if (lambdaInputItem > 0 || lambdaOutputItem > 0){
								double w1=0, w2=0;
								int worse_did = ((RankingDataset)trainSet.dataset).qdLines[begin+worseIdx];
								if(inputItemNeighbours != null && inputItemNeighbours.containsKey(worse_did)) w1 = inputItemNeighbours.get(worse_did);
								if(outputItemNeighbours != null && outputItemNeighbours.containsKey(worse_did)) w2 = outputItemNeighbours.get(worse_did);
								w = (lambdaInputItem * w1 + lambdaOutputItem * w2) / (lambdaInputItem + lambdaOutputItem);
							}
								
							pairWeight = (1-w) * (NDCGEval.GAINS[labels[betterIdx]] - NDCGEval.GAINS[labels[worseIdx]])
							* Math.abs((NDCGEval.discounts[i] - NDCGEval.discounts[j])) / queryMaxDcg;	
							
							//pairWeight = (1-w) * Math.abs((NDCGEval.GAINS[labels[betterIdx]] - NDCGEval.GAINS[labels[worseIdx]])
							//* (NDCGEval.discounts[i] - NDCGEval.discounts[j]) / queryMaxDcg);
								
							if (labels[betterIdx] > labels[worseIdx]) {
								
								scoreDiff = trainPredictions[begin + betterIdx] - trainPredictions[begin + worseIdx];
									
								if (scoreDiff <= minScore) {
									rho = sigmoidCache[0];
								} else if (scoreDiff >= maxScore) {
									rho = sigmoidCache[sigmoidCache.length - 1];
								} else {
									rho = sigmoidCache[(int) ((scoreDiff - minScore) / sigmoidBinWidth)];
								}
								
								double res = rho * pairWeight;
								
								gvi = gvi.plus(res,u);
								gvj = gvj.plus(-res,u);
								
								gu = gu.plus(res,vi).plus(-res,vj);
								
							}
							
							GV.insertIntoThis(djd, 0, gvj);
						
						}
						
						
						
						GV.insertIntoThis(did, 0, gvi);	
					}
					
					GU.insertIntoThis(query, 0, gu);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

}
