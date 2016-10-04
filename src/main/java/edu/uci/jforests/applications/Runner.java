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
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.TreeMap;

import org.ejml.simple.SimpleMatrix;

import joptsimple.OptionParser;
import joptsimple.OptionSet;
import edu.uci.jforests.config.TrainingConfig;
import edu.uci.jforests.dataset.Dataset;
import edu.uci.jforests.dataset.DatasetLoader;
import edu.uci.jforests.dataset.RankingDataset;
import edu.uci.jforests.dataset.RankingDatasetLoader;
import edu.uci.jforests.eval.ranking.NDCGEval;
import edu.uci.jforests.input.RankingRaw2BinConvertor;
import edu.uci.jforests.input.Raw2BinConvertor;
import edu.uci.jforests.learning.LearningUtils;
import edu.uci.jforests.learning.trees.Ensemble;
import edu.uci.jforests.learning.trees.decision.DecisionTree;
import edu.uci.jforests.learning.trees.regression.EnsembleMF;
import edu.uci.jforests.learning.trees.regression.RegressionTree;
import edu.uci.jforests.learning.trees.regression.RegressionTree_MF;
import edu.uci.jforests.sample.RankingMFSample;
import edu.uci.jforests.sample.RankingSample;
import edu.uci.jforests.sample.Sample;
import edu.uci.jforests.util.IOUtils;
import edu.uci.jforests.util.Util;
import edu.uci.jforests.util.concurrency.BlockingThreadPoolExecutor;

/**
 * @author Yasser Ganjisaffar <ganjisaffar at gmail dot com>
 */

public class Runner {

	@SuppressWarnings("unchecked")
	private static void generateBin(OptionSet options) throws Exception {
		if (!options.has("folder")) {
			System.err.println("The input folder is not specified.");
			return;
		}

		if (!options.has("file")) {
			System.err.println("Input files are not specified.");
			return;
		}

		String folder = (String) options.valueOf("folder");
		List<String> filesList = (List<String>) options.valuesOf("file");
		String[] files = new String[filesList.size()];
		for (int i = 0; i < files.length; i++) {
			files[i] = filesList.get(i);
		}

		if (options.has("ranking")) {
			System.out.println("Generating binary files for ranking data sets...");
			new RankingRaw2BinConvertor().convert(folder, files);
		} 
		else {
			System.out.println("Generating binary files...");
			new Raw2BinConvertor().convert(folder, files);
		}
	}

	private static void train(OptionSet options) throws Exception {
		if (!options.has("config-file")) {
			System.err.println("The configurations file is not specified.");
			return;
		}

		InputStream configInputStream = new FileInputStream((String) options.valueOf("config-file"));
		Properties configProperties = new Properties();
		configProperties.load(configInputStream);

		if (options.has("train-file")) {
			configProperties.put(TrainingConfig.TRAIN_FILENAME, options.valueOf("train-file"));
		}

		if (options.has("validation-file")) {
			configProperties.put(TrainingConfig.VALID_FILENAME, options.valueOf("validation-file"));
		}

		Ensemble ensemble;

		if (options.has("ranking")) {
			RankingApp app = new RankingApp();
			ensemble = app.run(configProperties);
		} 
		else if (options.has("ranking-mf")) {
			RankingMFApp app = new RankingMFApp();
			ensemble = app.run(configProperties);
		} else {
			ClassificationApp app = new ClassificationApp();
			ensemble = app.run(configProperties);
		}

		/*
		 * Dump the output model if requested.
		 */
		if (options.has("output-model")) {
			String outputModelFile = (String) options.valueOf("output-model");
			File file = new File(outputModelFile);
			PrintStream ensembleOutput = new PrintStream(file);
			ensembleOutput.println(ensemble);
			ensembleOutput.close();
		}

	}

	private static void predict(OptionSet options) throws Exception {

		if (!options.has("model-file")) {
			System.err.println("Model file is not specified.");
			return;
		}

		if (!options.has("tree-type") ) {
			System.err.println("Types of trees in the ensemble is not specified.");
			return;
		}

		if (!options.has("test-file")) {
			System.err.println("Test file is not specified.");
			return;
		}

		/*
		 * Load the ensemble
		 */
		File modelFile = new File((String) options.valueOf("model-file"));
		Ensemble ensemble = new Ensemble();
		if (options.valueOf("tree-type").equals("RegressionTree") && options.has("ranking-mf")) {
			ensemble = new EnsembleMF();
			ensemble.loadFromFile(RegressionTree_MF.class, modelFile);
		} else if (options.valueOf("tree-type").equals("RegressionTree")) {
			ensemble.loadFromFile(RegressionTree.class, modelFile);
		} else if (options.valueOf("tree-type").equals("DecisionTree")) {
			ensemble.loadFromFile(DecisionTree.class, modelFile);
		} else {
			System.err.println("Unknown tree type: " + options.valueOf("tree-type"));
		}

		/*
		 * Load the data set
		 */
		InputStream in = new IOUtils().getInputStream((String) options.valueOf("test-file"));
		Sample sample;
		if (!options.has("ranking") && !options.has("ranking-mf")){
			Dataset dataset = new Dataset();
			DatasetLoader.load(in, dataset);
			sample = new Sample(dataset);
		}
		else {
			
			RankingDataset dataset = new RankingDataset();
			RankingDatasetLoader.load(in, dataset);
			
			if (options.has("ranking-mf")) {
				
				if (!options.has("U")) {
					System.err.println("U file is not specified.");
					return;
				}
				
				if (!options.has("V")) {
					System.err.println("V file is not specified.");
					return;
				}
				
				if (!options.has("docIds")) {
					System.err.println("docIds file is not specified.");
					return;
				}
				
				sample = new RankingMFSample(dataset);
				
				String csvFileForU = (String) options.valueOf("U");
				String csvFileForV = (String) options.valueOf("V");
				String docIdsFile = (String) options.valueOf("docIds");
				//if(((EnsembleMF)ensemble).getNumTrees() == 0)
					((RankingMFSample)sample).loadFactors(csvFileForU, csvFileForV, docIdsFile, true, (EnsembleMF)ensemble);
				//else 
				//	((RankingMFSample)sample).loadFactors(csvFileForU, csvFileForV, docIdsFile, false);
		
			}
			else sample = new RankingSample(dataset);
				
			BlockingThreadPoolExecutor.init(1);
			NDCGEval.initialize(dataset.maxDocsPerQuery);
			int[][] labelCounts = NDCGEval.getLabelCountsForQueries(dataset.targets, dataset.queryBoundaries);
			dataset.maxDCG = NDCGEval.getMaxDCGForAllQueriesUptoTruncation(dataset.targets, dataset.queryBoundaries,
					NDCGEval.MAX_TRUNCATION_LEVEL, labelCounts);
			
		} 
		in.close();
		
		double[] predictions = new double[sample.size];
		PrintStream output;
		if (options.has("output-file")) {
			output = new PrintStream(new File((String) options.valueOf("output-file")));
		} else {
			output = System.out;
		}
	
		if (options.has("ranking") || options.has("ranking-mf")) {
			
			if (!options.has("evalTrunc")) {
				System.err.println("int evalTrunc for NDCG is not specified.");
				return;
			}
			
			if (options.has("ranking")) LearningUtils.updateScores(sample, predictions, ensemble);
			else {
				
				if (!options.has("learningRate")) {
					System.err.println("double learningRate for gradient trees is not specified.");
					return;
				}
				
				double learningRate = Double.parseDouble((String) options.valueOf("learningRate"));
				LearningUtils.updateScores((RankingMFSample)sample, predictions, (EnsembleMF)ensemble, learningRate);
				
				((RankingMFSample)sample).U.saveToFileCSV("U.test.csv");
				((RankingMFSample)sample).V.saveToFileCSV("V.test.csv");
			}
		
			int evalTruncation = Integer.parseInt((String) options.valueOf("evalTrunc"));
			NDCGEval evalMetric = new NDCGEval(((RankingDataset)sample.dataset).maxDocsPerQuery, evalTruncation);
			double v = sample.evaluate(predictions, evalMetric);
			output.println(v);
			
			//for (int i = 0; i < sample.size; i++) {
			//	output.println(predictions[i]);
			//}
			
			for(int i=0; i < evalMetric.ndcgs.length; i++){
				output.print(i+"="+evalMetric.ndcgs[i]+" ");
				
				int begin = ((RankingSample)sample).queryBoundaries[i];
				int numDocuments = ((RankingSample)sample).queryBoundaries[i + 1] - begin;
				
				for (int j=0; j<numDocuments; j++){
					
					// The item index among other items
					int idx = ((RankingDataset)sample.dataset).qdLines[begin+j];
					output.print(idx+"="+predictions[begin + j]+" ");
				}
				
				output.println();
			}
			
			
		} else {
			
			LearningUtils.updateScores(sample, predictions, ensemble);
		
			for (int i = 0; i < sample.size; i++) {
				output.println(predictions[i]);
			}
		}
	}
	
	private static void kernelize(OptionSet options) throws Exception {
		
		if (!options.has("nn")) {
			System.err.println("Number of nearest neighbours is not specified.");
			return;
		}
		
		// How to compute linearly average euclidean distance of the m matrix ?
		if (!options.has("mean-distance")) {
			System.err.println("Estimated mean euclidean distance is not specified.");
			return;
		}
		
		if (!options.has("file1") && !options.has("file2")) {
			System.err.println("Input data file1 and file2 are not specified.");
			return;
		}
		
		int nn = Integer.parseInt((String)options.valueOf("nn"));
		double mean_dist = Double.parseDouble((String)options.valueOf("mean-distance"));
		SimpleMatrix m1 = SimpleMatrix.loadCSV((String) options.valueOf("file1"));
		SimpleMatrix m2 = SimpleMatrix.loadCSV((String) options.valueOf("file2"));
		m1.printDimensions();
		m2.printDimensions();

		if(m1.numCols() != m2.numCols()) throw(new Exception("file1 and file2 have to be of same dimensionality !"));
			
		// Compute mean and sd from the two data files
		double[] means = new double[m1.numCols()]; 
		double[] sd = new double[m1.numCols()]; 
		
		for(int i=0; i<m1.numCols(); i++){
			SimpleMatrix v1 = m1.extractVector(false, i);
			SimpleMatrix v2 = m2.extractVector(false, i);
			means[i] = 0.5 * (v1.elementSum() / m1.numRows() + v2.elementSum() / m2.numRows());
			double sq = (v1.elementMult(v1).elementSum() + v2.elementMult(v2).elementSum());
			sd[i] = Math.sqrt( (sq / (m1.numRows() + m2.numRows())) - Math.pow(means[i], 2) );
		}
		
		// Build and store m1 data
		double[][] keys1 = new double[m1.numRows()][m1.numCols()-1]; 
		double[] value1 = new double[m1.numRows()];
		
		for(int i=0; i<m1.numRows(); i++){
			SimpleMatrix v = m1.extractVector(true, i);
			double[] data = v.getMatrix().getData();
			// Standardize data
			for(int j=1; j<m1.numCols(); j++) if(sd[j]>0) data[j] = (data[j]-means[j])/sd[j];
			value1[i] = data[0];
			System.arraycopy(data, 1, keys1[i], 0, keys1[i].length);
		}
		
		// Build and store m2 data
		double[][] keys2 = new double[m2.numRows()][m2.numCols()-1]; 
		double[] value2 = new double[m2.numRows()];
		
		for(int i=0; i<m2.numRows(); i++){
			SimpleMatrix v = m2.extractVector(true, i);
			double[] data = v.getMatrix().getData();
			// Standardize data
			for(int j=1; j<m2.numCols(); j++) if(sd[j]>0) data[j] = (data[j]-means[j])/sd[j];
			value2[i] = data[0];
			System.arraycopy(data, 1, keys2[i], 0, keys2[i].length);
		}
		
		// Retrieve similarities previously stored in order to update the top-n
		TreeMap<Integer,String> tm = new TreeMap<Integer,String>();
		File f = new File(((String) options.valueOf("file1")) + "." +nn + ".nn");
		try{
			FileInputStream fstream = new FileInputStream(f.getAbsolutePath());
			BufferedReader br = new BufferedReader(new InputStreamReader(fstream));
			
			for(String line; (line = br.readLine()) != null; )  {
				String[] tokens = line.split(": ");
				tm.put(Integer.parseInt(tokens[0]), tokens[1]);
			}
			br.close();
		} catch(IOException e){
			f.createNewFile();
		}
		
		FileWriter fileWritter = new FileWriter(f.getAbsolutePath(),false);
        BufferedWriter bufferWritter = new BufferedWriter(fileWritter);
        
        TreeMap<Double,Integer> tnn = new TreeMap<Double,Integer>(Collections.reverseOrder());
        double sigma = 1/(mean_dist*mean_dist);
		
		// Compute similarities and keep track only of top-n
        for(int i=0; i<value1.length; i++){
        	int idx = (int)value1[i];
			
        	if(i % 100 == 0) System.out.println(i); // Log output for the progression
			tnn.clear();
			
			// Fill up tnn with past similarities
			if(tm.containsKey(idx)){
				String line = tm.get(idx);
				String[] tokens = line.split(" ");
				for(int t=0; t<tokens.length; t++){
					String[] pair = tokens[t].split("=");
					tnn.put(Double.parseDouble(pair[1]), Integer.parseInt(pair[0]));
				}
			}
			
			// Compute new similarities, feeds them into tnn and keep only the n bests
			for(int j=0; j<value2.length; j++){
				if(i==j) continue;
				double dist = sqrdist(keys1[i], keys2[j]);
				tnn.put(Math.exp(-sigma * dist) ,(int)value2[j]);
	    		if (tnn.size() > nn) tnn.remove(tnn.lastKey());
			}
			
			bufferWritter.write(idx+": ");
			for(Double key: tnn.keySet()) { 
				bufferWritter.write(tnn.get(key)+"="+key+" ");
				//System.out.println("\t"+tnn.get(key)+"="+key);
			}
			
			bufferWritter.write("\n");
			bufferWritter.flush();
			
			//System.out.println();
			//Util.readline();
		}
		
		bufferWritter.close();
		fileWritter.close();

	}
	
	public static double sqrdist(double [] a, double [] b) {

		double dist = 0;

		for (int i=0; i<a.length; ++i) {
		    double diff = (a[i] - b[i]);
		    dist += diff*diff;
		}

		return dist;
	    } 
	
	public static void main(String[] args) throws Exception {

		OptionParser parser = new OptionParser();

		parser.accepts("cmd").withRequiredArg();
		parser.accepts("ranking");
		parser.accepts("ranking-mf");
		
		/*
		 * Bin generation arguments
		 */
		parser.accepts("folder").withRequiredArg();
		parser.accepts("file").withRequiredArg();

		/*
		 * Training arguments
		 */
		parser.accepts("config-file").withRequiredArg();
		parser.accepts("train-file").withRequiredArg();
		parser.accepts("validation-file").withRequiredArg();
		parser.accepts("output-model").withRequiredArg();

		/*
		 * Prediction arguments
		 */
		parser.accepts("model-file").withRequiredArg();
		parser.accepts("tree-type").withRequiredArg();
		parser.accepts("test-file").withRequiredArg();
		parser.accepts("output-file").withRequiredArg();

		// Arguments for ranking-mf
		parser.accepts("U").withRequiredArg(); // the U training factors
		parser.accepts("V").withRequiredArg(); // the V training factors
		parser.accepts("docIds").withRequiredArg(); // training doc ids 
		parser.accepts("evalTrunc").withRequiredArg(); // ndcg truncation level
		parser.accepts("learningRate").withRequiredArg(); // training learning rate with which we will update the testing factors
		/*
		 * Kernel computation arguments
		 */
		parser.accepts("file1").withRequiredArg();
		parser.accepts("file2").withRequiredArg();
		parser.accepts("nn").withRequiredArg();
		parser.accepts("mean-distance").withRequiredArg();
		
		OptionSet options = parser.parse(args);

		if (!options.has("cmd")) {
			System.err.println("You must specify the command through 'cmd' parameter.");
			return;
		}

		if (options.valueOf("cmd").equals("generate-bin")) {
			generateBin(options);
		} else if (options.valueOf("cmd").equals("train")) {
			train(options);
		} else if (options.valueOf("cmd").equals("predict")) {
			predict(options);
		} else if (options.valueOf("cmd").equals("kernelize")) {
			kernelize(options);
		} else {
			System.err.println("Unknown command: " + options.valueOf("cmd"));
		}

		/*
		 * Make sure that thread pool is terminated.
		 */
		ClassificationApp.shutdown();
	}
}
