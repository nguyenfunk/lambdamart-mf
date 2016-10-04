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

package edu.uci.jforests.input;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;
import java.util.TreeSet;

import edu.uci.jforests.input.sparse.SparseTextFileLine;
import edu.uci.jforests.util.ArraysUtil;
import edu.uci.jforests.util.Util;

/**
 * @author Yasser Ganjisaffar <ganjisaffar at gmail dot com>
 */

public class RankingBinFileGenerator extends BinaryFileGenerator {

	private String prevQid;
	private int lineIndex;
	protected List<Integer> queryBoundaries;
	
	protected List<Integer> queryIdx;
	protected TreeSet<Integer> docIdx;
	protected TreeMap<Integer,Integer> docBoundaries;
	protected List<Integer> qdLines;
	
	public RankingBinFileGenerator(String textFile, String featuresStatFile, String binFile) {
		super(textFile, featuresStatFile, binFile);
		prevQid = null;
		lineIndex = 0;
		queryBoundaries = new ArrayList<Integer>();
		
		queryIdx = new ArrayList<Integer>();
		docIdx = new TreeSet<Integer>();
		docBoundaries = new TreeMap<Integer,Integer>();
		qdLines = new ArrayList<Integer>();
	}
	
	@Override
	protected void handle(SparseTextFileLine line) {
		if (!line.qid.equals(prevQid)) {
			queryBoundaries.add(lineIndex);
			queryIdx.add(Integer.parseInt(line.qid));
		}
		
		if(line.did != null){
			
			Integer docId = Integer.parseInt(line.did);
			// Sort docIds in a TreeSet 
			if(docIdx.add(docId)) docBoundaries.put(docId, lineIndex);	
			// For each line, store its docId
			qdLines.add(docId);
		}
		
		prevQid = line.qid;
		lineIndex++;
	}

	@Override
	protected void loadValueHashMaps() {
		super.loadValueHashMaps();
		queryBoundaries.add(lineIndex);
	}
	
	@Override
	protected void createBinFile() {
		writer = new RankingBinFileWriter(binFile, features, targets, 
				ArraysUtil.toArray(queryBoundaries), 
				ArraysUtil.toArray(queryIdx), 
				ArraysUtil.toArray(new ArrayList<Integer>(this.docIdx)), 
				ArraysUtil.toArray(new ArrayList<Integer>(this.docBoundaries.values())),
				ArraysUtil.toArray(qdLines)
		);
	}
	
}
