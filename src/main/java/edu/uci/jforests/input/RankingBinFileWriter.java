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

import edu.uci.jforests.dataset.Feature;

/**
 * @author Yasser Ganjisaffar <ganjisaffar at gmail dot com>
 */

public class RankingBinFileWriter extends BinaryFileWriter {

	private int[] queryBoundaries, docBoundaries;
	private int[] queryIdx, docIdx, qdLines;
	
	public RankingBinFileWriter(String binFileName, Feature[] features, double[] targets, int[] queryBoundaries, int[] queryIdx, int[] docIdx, int[] docBoundaries, int[] qdLines) {
		super(binFileName, features, targets);
		this.queryBoundaries = queryBoundaries;
		this.queryIdx = queryIdx;
		this.docIdx = docIdx;	
		this.docBoundaries = docBoundaries;
		this.qdLines = qdLines;
	}

	@Override
	protected void writeHeader() {
		super.writeHeader();
		int numQueries = queryBoundaries.length - 1;
		writeInt(numQueries);
		writeInt(queryIdx.length);
		writeInt(docIdx.length);
		writeInt(docBoundaries.length);
		writeInt(qdLines.length);
	}
	
	@Override
	public void write() {
		super.write();
		for (int q = 0; q < queryBoundaries.length; q++) {
			writeInt(queryBoundaries[q]);			
		}
		for (int q = 0; q < queryIdx.length; q++) {
			writeInt(queryIdx[q]);			
		}
		for (int q = 0; q < docIdx.length; q++) {
			writeInt(docIdx[q]);			
		}
		for (int q = 0; q < docBoundaries.length; q++) {
			writeInt(docBoundaries[q]);			
		}
		for (int q = 0; q < qdLines.length; q++) {
			writeInt(qdLines[q]);			
		}
	}
}
