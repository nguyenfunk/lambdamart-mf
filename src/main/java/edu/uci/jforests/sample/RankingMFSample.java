/**
 * 
 */
package edu.uci.jforests.sample;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;
import org.ejml.simple.SimpleMatrix;

/*
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
*/

import edu.uci.jforests.dataset.RankingDataset;
import edu.uci.jforests.util.ArraysUtil;
import edu.uci.jforests.util.IOUtils;

import edu.uci.jforests.learning.trees.regression.EnsembleMF;
import edu.uci.jforests.learning.trees.regression.RegressionTree_MF;


/**
 * @author phong_nguyen
 *
 */
public class RankingMFSample extends RankingSample {

	public int[] queryIdx, userIndicesInDataset, itemIndicesInDataset;
	public SimpleMatrix U, V, target;
	//public DoubleMatrix2D U, V; 
	public int numDocs;
	
	/**
	 * @param dataset
	 */
	public RankingMFSample(RankingDataset dataset) {
		super(dataset);
		queryIdx = dataset.queryIdx;
		
		userIndicesInDataset = new int[numQueries];
		for(int i=0; i<numQueries;i++){
			int instanceIdx = dataset.queryBoundaries[i];
			userIndicesInDataset[i] = instanceIdx;
		}
		
		numDocs = dataset.docIdx.size();
		itemIndicesInDataset = new int[numDocs];
		
		for(int i=0; i<numDocs;i++){
			int instanceIdx = dataset.docBoundaries[i];
			itemIndicesInDataset[i] = instanceIdx;
		}
	}

	/**
	 * @param dataset
	 * @param queryIndices
	 * @param queryBoundaries
	 * @param queryIdx
	 * @param instances
	 * @param weights
	 * @param targets
	 * @param indicesInParentSample
	 * @param docCount
	 * @param queryCount
	 * @param U
	 * @param V
	 */
	public RankingMFSample(RankingDataset dataset, int[] queryIndices,
			int[] queryBoundaries, int[] queryIdx, int[] instances,
			double[] weights, double[] targets, int[] indicesInParentSample,
			int docCount, int queryCount, SimpleMatrix U, SimpleMatrix V, 
			int[] userIndicesInDataset, int[] itemIndicesInDataset) {
			//int docCount, int queryCount, DoubleMatrix2D U, DoubleMatrix2D V) {
		super(dataset, queryIndices, queryBoundaries,  instances,
				weights, targets, indicesInParentSample, docCount, queryCount);
		
		this.U = U;
		this.V = V;
		this.userIndicesInDataset = userIndicesInDataset;
		this.itemIndicesInDataset = itemIndicesInDataset;
		this.queryIdx = queryIdx;
		this.numQueries = userIndicesInDataset.length;
		this.numDocs = itemIndicesInDataset.length;
	}
	
	public void initFactors(int nb_factors, Random rnd){
		
		this.U = SimpleMatrix.random(numQueries, nb_factors, -1, 1, rnd);
		this.V = SimpleMatrix.random(numDocs, nb_factors, -1, 1, rnd);
		
		//this.U.set(1);
		//this.V.set(1);
		
		//this.U = new SparseDoubleMatrix2D(numQueries, nb_factors);
		//this.U.assign(1);
		//DoubleFactory2D f = DoubleFactory2D.dense;
		//this.V = f.random(numDocs, nb_factors);
		
		System.out.println("Factors initialized.");
		U.printDimensions();
		V.printDimensions();
	}
	
	public void loadFactors(String csvFileForU, String csvFileForV, String docIdsFile, boolean copyU, EnsembleMF ensemble){
		try {
			SimpleMatrix trainU = SimpleMatrix.loadCSV(csvFileForU);
			SimpleMatrix trainV = SimpleMatrix.loadCSV(csvFileForV);
			
			initFactors(trainU.numCols(), new Random());
			if (copyU) this.U = trainU.copy();
			
			IOUtils ioUtils = new IOUtils();
			InputStream in = ioUtils.getInputStream(docIdsFile);
			BufferedReader reader = new BufferedReader(new InputStreamReader(in));
			String [] tokens = reader.readLine().split(", ");
		    
			TreeMap<Integer,Integer> docIdx = new TreeMap<Integer,Integer>();
		    for (int i=0; i<tokens.length; i++){
		    	Integer idx = Integer.parseInt(tokens[i]);
		    	docIdx.put(idx, i);
		    }
		    
		    TreeSet<Integer> intersection = new TreeSet<Integer>(((RankingDataset)this.dataset).docIdx.keySet());
		    intersection.retainAll(docIdx.keySet());
		    System.out.println("V intersection size = "+intersection.size());
			for(int i : intersection){
				int tidx = docIdx.get(i);
				int vidx = ((RankingDataset)this.dataset).docIdx.get(i);
				SimpleMatrix v = trainV.extractVector(true, tidx);
				this.V.insertIntoThis(vidx, 0, v);
			}
			
			double[] ones = new double[trainV.numRows()];
			for(int i=0; i<trainV.numRows(); i++) ones[i] = 1.0000;
			
			SimpleMatrix v1 = new SimpleMatrix(trainV.numRows(), 1);
			v1.setColumn(0, 0, ones);
			SimpleMatrix vmean = trainV.transpose().mult(v1).divide(this.V.numRows());
			
			TreeSet<Integer> diff = new TreeSet<Integer>(((RankingDataset)this.dataset).docIdx.keySet());
		    diff.removeAll(docIdx.keySet());
		    System.out.println("V diff size = "+diff.size() + ", num trees="+ensemble.getNumTrees());
		    for(int i : diff){
		    	int vidx = ((RankingDataset)this.dataset).docIdx.get(i);
		    	SimpleMatrix v = this.V.extractVector(true, vidx);
		    	for (int t = 0; t < ensemble.getNumTrees(); t++) {
		    		RegressionTree_MF itemTree = (RegressionTree_MF) ensemble.getItemTreeAt(t);
		    		SimpleMatrix g =  itemTree.getMultiOutput(this.dataset, this.itemIndicesInDataset[vidx]);
		    		v = v.plus(0.01, g);
		    	}
		    	this.V.setRow(vidx, 0, v.getMatrix().data);
		    }
			
				// Look for the intersection between training and testing items 
		    /*
			TreeMap<Integer,Integer> docIdx = ((RankingDataset)this.dataset).docIdx;
			int intersection_size = 0;
			for (int i=0; i<tokens.length; i++){
				int idx = Integer.parseInt(tokens[i]);
				
				// If an item is used both for training and testing, copy in the V test matrix its training value
				if(docIdx.containsKey(idx)){
					int vidx = docIdx.get(idx);
					SimpleMatrix v = trainV.extractVector(true, i);
					this.V.insertIntoThis(vidx, 0, v);
					intersection_size++;
				} 
				
				// If not, assign the factor's mean
				//else {
				//	//this.V.setRow(vidx, 0, vmean.getMatrix().data);
				//	this.V.insertIntoThis(vidx, 0, new SimpleMatrix(trainV.numCols(), 0));
				//}
			}
			
			System.out.println("V intersection: "+intersection_size);
			*/
			
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	@Override
	public RankingMFSample getClone() {
		return new RankingMFSample((RankingDataset) dataset, queryIndices, queryBoundaries, queryIdx, indicesInDataset, weights, targets, indicesInParentSample, size,
				numQueries, U, V, userIndicesInDataset, itemIndicesInDataset);
	}

	public RankingMFSample getRandomUserSubSample(double rate, Random rnd) {
		if (rate < 1.0) {
			int subSampleNumQueries = (int) (numQueries * rate);
			int[] tempQueryIndices = new int[numQueries];
			for (int i = 0; i < numQueries; i++) {
				tempQueryIndices[i] = i;
			}
			ArraysUtil.shuffle(tempQueryIndices, rnd);
			Arrays.sort(tempQueryIndices, 0, subSampleNumQueries);
			int[] subSampleQueryBoundaries = new int[subSampleNumQueries + 1];
			int subSampleSize = 0;
			for (int idx = 0; idx < subSampleNumQueries; idx++) {
				int f = tempQueryIndices[idx];
				subSampleSize += queryBoundaries[f + 1] - queryBoundaries[f];
			}
			int[] sampleIndicesInDataset = new int[subSampleSize];	
			double[] sampleWeights = new double[subSampleSize];
			double[] sampleTargets = new double[subSampleSize];
			int[] sampleIndicesInParentSample = new int[subSampleSize];
			int curSampleSize = 0;
			int[] subSampleQueryIndices = new int[subSampleNumQueries];
			int[] subSampleQueryIdx = new int[subSampleNumQueries];
			int[] subSampleUserIndicesInDataset = new int[subSampleNumQueries];
			
			for (int idx = 0; idx < subSampleNumQueries; idx++) {
				int f = tempQueryIndices[idx];
				subSampleQueryBoundaries[idx] = curSampleSize;
				int beginOffset = queryBoundaries[f];
				int numDocs = queryBoundaries[f + 1] - beginOffset;
				for (int d = 0; d < numDocs; d++) {
					sampleIndicesInDataset[curSampleSize] = indicesInDataset[beginOffset + d];
					sampleWeights[curSampleSize] = weights[beginOffset + d];
					sampleTargets[curSampleSize] = targets[beginOffset + d];
					sampleIndicesInParentSample[curSampleSize] = beginOffset + d;
					curSampleSize++;
				}
				subSampleQueryIndices[idx] = queryIndices[f];
				subSampleQueryIdx[idx] = queryIdx[f];
				subSampleUserIndicesInDataset[idx] = userIndicesInDataset[f];
			}
			
			/*
			SimpleMatrix Ucopy = null;
			//SparseDoubleMatrix2D Ucopy = null;
			if(U!=null){
				Ucopy = new SimpleMatrix(subSampleNumQueries, U.numCols());
				//Ucopy = new SparseDoubleMatrix2D(subSampleNumQueries, U.columns());
				
				for(int i=0; i<subSampleNumQueries; i++){
					Ucopy.insertIntoThis(i, 0, U.extractVector(true, subSampleQueryIndices[i]));
					//Ucopy.viewRow(i).assign(U.viewRow(subSampleQueryIndices[i]));
				}
			} 
			*/
			
			subSampleQueryBoundaries[subSampleNumQueries] = curSampleSize;
			return new RankingMFSample((RankingDataset) dataset, subSampleQueryIndices, subSampleQueryBoundaries, subSampleQueryIdx, sampleIndicesInDataset, sampleWeights,
					sampleTargets, sampleIndicesInParentSample, subSampleSize, subSampleNumQueries, U, V, subSampleUserIndicesInDataset, itemIndicesInDataset);
		} else {
			RankingMFSample result = this.getClone();
			//result.indicesInParentSample = Constants.ONE_TWO_THREE_ETC;
			return result;
		}
	}
	
	public RankingMFSample getRandomItemSubSample(double rate, Random rnd) {
		if (rate < 1.0) {
			int subSampleNumDocs = (int) (numDocs * rate);
			int[] tempDocIndices = new int[numDocs];
			for (int i = 0; i < numDocs; i++) {
				tempDocIndices[i] = i;
			}
			ArraysUtil.shuffle(tempDocIndices, rnd);
			Arrays.sort(tempDocIndices, 0, subSampleNumDocs);
			
			int[] subSampleItemIndicesInDataset = new int[subSampleNumDocs];
			
			for (int idx = 0; idx < subSampleNumDocs; idx++) {
				int f = tempDocIndices[idx];
				subSampleItemIndicesInDataset[idx] = itemIndicesInDataset[f];
			}
			
			return new RankingMFSample((RankingDataset) dataset, queryIndices, queryBoundaries, queryIdx, indicesInDataset, weights, targets, indicesInParentSample, size,
					numQueries, U, V, userIndicesInDataset, subSampleItemIndicesInDataset);
		}
		else return getClone();
	}
	
	public void printDimensions(){
		System.out.println("RankingMFSample[numQueries:"+numQueries+", numDocs:"+numDocs+"]");
	}
}
