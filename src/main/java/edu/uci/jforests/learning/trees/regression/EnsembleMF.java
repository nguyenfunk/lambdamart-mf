/**
 * 
 */
package edu.uci.jforests.learning.trees.regression;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import edu.uci.jforests.learning.trees.Ensemble;
import edu.uci.jforests.learning.trees.Tree;
import edu.uci.jforests.util.Util;

/**
 * @author phong_nguyen
 *
 */
public class EnsembleMF extends Ensemble {

	private List<Tree> itemTrees;
	private List<Double> itemWeights;
	private int nb_factors;
	
	public EnsembleMF() {
		super();
		itemTrees = new ArrayList<Tree>();
		itemWeights = new ArrayList<Double>();
	}

	public List<Tree> getItemTrees() {
		return itemTrees;
	}

	public void addItemTree(Tree tree, double weight) {
		itemTrees.add(tree);
		itemWeights.add(weight);
	}

	public void addItemTreeAt(Tree tree, double weight, int index) {
		itemTrees.add(index, tree);
		itemWeights.add(weight);
	}

	public void removeItemTree(int index) {
		itemTrees.remove(index);
		itemWeights.remove(index);
	}

	public void removeLastItemTrees(int k) {
		for (int i = 0; i < k; i++) {
			removeItemTree(itemTrees.size() - 1);
		}
	}

	public Tree getItemTreeAt(int index) {
		return itemTrees.get(index);
	}

	public double getItemWeightAt(int index) {
		return itemWeights.get(index);
	}

	public int getNumItemTrees() {
		return itemTrees.size();
	}

	public List<Double> getItemWeights() {
		return itemWeights;
	}

	public void setItemWeights(List<Double> weights) {
		this.itemWeights = weights;
	}
	
	public String toString(int prefix) {
		if (prefix > trees.size() || prefix < 0) {
			prefix = trees.size();
		}
		StringBuilder sb = new StringBuilder();
		sb.append("<Ensemble>");
		for (int i = 0; i < prefix; i++) {
			sb.append(trees.get(i).toString(weights.get(i), 1, "user", ((RegressionTree_MF) trees.get(i)).nb_factors ));
		}

		if (prefix > itemTrees.size() || prefix < 0) {
			prefix = itemTrees.size();
		}

		for (int i = 0; i < prefix; i++) {
			sb.append(itemTrees.get(i).toString(itemWeights.get(i), 1, "item", ((RegressionTree_MF) itemTrees.get(i)).nb_factors));
		}
		sb.append("\n</Ensemble>");
		return sb.toString();
	}
	
	@Override
	public <T extends Tree> void loadFromFile(Class<T> _c, File file) throws Exception {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		String line = reader.readLine(); // <Ensemble>
		while ((line = reader.readLine()) != null) {
			line = line.trim();
			if (line.equals("</Ensemble>")) {
				break;
			}
			String header = line;
			int leaves = Integer.parseInt(getXmlAttribute(header, "leaves"));
			double weight = Double.parseDouble(getXmlAttribute(header, "weight"));
			String type = getXmlAttribute(header, "type");
			int factors = Integer.parseInt(getXmlAttribute(header, "factors"));
			
			String featuresLine = reader.readLine();
			String leftChildrenLine = reader.readLine();
			String rightChildrenLine = reader.readLine();
			String thresholds = reader.readLine();
			String originalThresholds = reader.readLine();

			T tree = _c.newInstance();
			((RegressionTree_MF)tree).loadFromString(leaves, featuresLine, leftChildrenLine, rightChildrenLine, thresholds, originalThresholds, factors);

			for(int i=0; i<leaves; i++){
				((RegressionTree_MF)tree).loadCustomData(reader.readLine(), i);
			}
			
			reader.readLine(); // </RegressionTree>
			
			if (type.equals("user")) addTree(tree, weight);
			else addItemTree(tree, weight);
		}
		
		reader.close();
	}
}
