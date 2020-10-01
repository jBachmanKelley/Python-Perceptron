import java.io.*;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.HashMap;
import java.util.Map;

public class a2 {
	private String[][] data = new String[10000][11];
	private String[][] perfect = new String[10000][11];
	private float entries  = 0;
	private HashMap< Integer, Integer> editedRows = new HashMap< Integer, Integer>(); //(Row,Col)
	
	public static void main(String[] args) {
		// Import CSV files
		a2 mean_01 = new a2();
		mean_01.read("dataset_missing01.csv");
		mean_01.mean_imputation(0);
		mean_01.save("V00815554_missing01_imputed_mean.csv");
		System.out.println("MAE_01_mean = " + mean_01.verify());
		
		a2 cond_mean_01 = new a2();
		cond_mean_01.read("dataset_missing01.csv");
		cond_mean_01.mean_imputation(1);
		cond_mean_01.save("V00815554_missing01_imputed_mean_conditional.csv");
		System.out.println("MAE_01_mean_conditional = " + cond_mean_01.verify());
		
		a2 hd_01 = new a2();
		hd_01.read("dataset_missing01.csv");
		hd_01.hot_deck(0);
		hd_01.save("V00815554_missing01_imputed_hd.csv");
		System.out.println("MAE_01_hd = " + hd_01.verify());
		
		a2 cond_hd_01 = new a2();
		cond_hd_01.read("dataset_missing01.csv");
		cond_hd_01.mean_imputation(1);
		cond_hd_01.save("V00815554_missing01_imputed_hd_conditional.csv");
		System.out.println("MAE_01_hd_conditional = " + cond_hd_01.verify());
		
		a2 mean_20 = new a2();
		mean_20.read("dataset_missing20.csv");
		mean_20.mean_imputation(0);
		mean_20.save("V00815554_missing20_imputed_mean.csv");
		System.out.println("MAE_20_mean = " + mean_20.verify());
		
		a2 cond_mean_20 = new a2();
		cond_mean_20.read("dataset_missing20.csv");
		cond_mean_20.mean_imputation(1);
		cond_mean_20.save("V00815554_missing20_imputed_mean_conditional.csv");
		System.out.println("MAE_20_mean_conditional = " + cond_mean_20.verify());
		
		a2 hd_20 = new a2();
		hd_20.read("dataset_missing20.csv");
		hd_20.hot_deck(0);
		hd_20.save("V00815554_missing20_imputed_hd.csv");
		System.out.println("MAE_20_hd = " + hd_20.verify());
		
		a2 cond_hd_20 = new a2();
		cond_hd_20.read("dataset_missing20.csv");
		cond_hd_20.mean_imputation(1);
		cond_hd_20.save("V00815554_missing20_imputed_hd_conditional.csv");
		System.out.println("MAE_20_hd_conditional = " + cond_hd_20.verify());
		
		
	}

	a2(){
		try {
			BufferedReader br = new BufferedReader(new FileReader(new File("dataset_complete.csv")));
			String line = "";
			String[] record;
			int i = 0, j = 0;
			while ((line = br.readLine()) != null) {
				j = 0;
				record = line.split(",");
				for (String attribute : record) {
					perfect[i][j] = attribute;
					j++;
				}
				i++;
			}
			br.close();
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
	}
	public void read(String csvFile) {
		try {
			BufferedReader br = new BufferedReader(new FileReader(new File(csvFile)));
			String line = "";
			String[] record;
			int i = 0, j = 0;
			while ((line = br.readLine()) != null) {
				j = 0;
				entries++;
				record = line.split(",");
				for (String attribute : record) {
					data[i][j] = attribute;
					j++;
				}
				i++;
			}
			br.close();
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
	}

	// int mode: 0 for nonconditional, 1 for conditional 
	public void mean_imputation(int mode) {
		if (mode == 1) {
			mean_imputation("Yes"); 
			mean_imputation("No");
		} else {
			mean_imputation("");
		}
	}
	public void mean_imputation(String conditional) {
		// Array from Sum and Counter Values, (Sum,Counter)
		double[][] temp = new double[2][10];
		for (int i = 1; i < entries; i++) 
			for (int j = 0; j < 10; j++) 
				if (!data[i][j].contains("?") && data[i][10].contains(conditional)) {
					temp[0][j] += Double.parseDouble(data[i][j]);
					temp[1][j]++;
				}			
		for (int i = 1; i < entries; i++)
			for (int j = 0; j < 10; j++) 
				if (data[i][j].contains("?") && data[i][10].contains(conditional)) {
					data[i][j] = "" + Math.round((temp[0][j] / temp[1][j]) * 100000.00) / 100000.00;
					editedRows.put((Integer)i, (Integer)j);
				}
	}

	public void hot_deck(int mode) {
		if (mode == 1) {
			hot_deck("Yes"); 
			hot_deck("No");
		} else {
			hot_deck("");
		}
	}
	public void hot_deck(String conditional) {
		// Go through checking for "?"
		// If found, call Manhattan Cycle, pass targetCol, and replace value
		for	(int i = 0; i < entries; i++) {
			for (int j = 0; j < 10; j++) {
				if(data[i][j].contains("?") && data[i][10].contains(conditional)) {
					editedRows.put( (Integer)i, (Integer)j);
					int k = this.manhattanCycle(i, j, conditional);
					//System.out.println("Problem Row: " + i + "Result Row: " + k);
					data[i][j] = data[k][j];
				}
			}
		}
	}
	
	public int manhattanCycle(int entryID, int targetCol, String conditional) {
		// Create a Structure
		ValueComparator vc = new ValueComparator();
		PriorityQueue<myNode> result = new PriorityQueue<myNode>((int)entries, vc);
		
		// Calculate distance IF HAS targetCol, HAS CONDITIONAL, and NOT Edited
		for (int i = 1; i < entries; i++) {
			if (!data[i][targetCol].contains("?") && !editedRows.containsKey((Integer)i) && data[i][10].contains(conditional))
				result.add(new myNode(i, calcD(entryID,i)));
		}
		
		// Return head myNode.ID
		return result.remove().entryID;
		
	}
	
	public float calcD(int entry1, int entry2) {
		float sum = 0;
		float counter = 0;
		
		for(int i = 0; i < 10; i++) {
			if(!data[entry1][i].contains("?") && !data[entry2][i].contains("?")) {
				sum += Math.abs(Float.parseFloat(data[entry1][i]) - Float.parseFloat(data[entry2][i]));
				counter++;
			}
		}
		return sum/counter;
	}
	
	// Computes MAE value and returns it
	public float verify() {
		float sum = 0;
		float counter = 0;
		// For each instance of an imputed value, compute difference and increment counter
		for(Map.Entry<Integer, Integer> entry : editedRows.entrySet()) {
			sum += Math.abs(Float.parseFloat(data[ entry.getKey()][ entry.getValue()]) - Float.parseFloat(perfect[ entry.getKey()][ entry.getValue()])) ;
			counter++;
		}
		
		return ((Math.round((sum/counter) * (float)10000.00)/(float)10000.00));
	}
	
	public void save(String target) {
		File file = new File(target);
		try (PrintWriter writer = new PrintWriter(file)) {
		      StringBuilder sb = new StringBuilder();
		      for( int i = 0; i < entries; i++) {
		    	  for( int j = 0; j < 10; j++) {
		    		  sb.append(data[i][j] + ",");
		    	  }
		    	  sb.append(data[i][10] + '\n');
		      }
		      writer.write(sb.toString());
		      
		    } catch (FileNotFoundException e) {
		      System.out.println(e.getMessage());
		    }
		if (file.exists())
			System.out.print("File Saved");
	}

	class myNode {
		int entryID;
		float distance;
		
		public myNode(int entryID, float distance){
			this.entryID = entryID;
			this.distance = distance;
		}
	}
	
	class ValueComparator implements Comparator<myNode> {

		@Override
		public int compare(myNode o1, myNode o2) {
			// TODO Auto-generated method stub
			return Float.compare(o1.distance, o2.distance);
		}
		
	}

}
