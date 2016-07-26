package Utils;

import it.unimi.dsi.fastutil.ints.IntArrays;
import math.DenseMap;
import math.SparseVector;

import java.io.*;
import java.util.*;

/**
 * Created by leleyu on 2016/6/30.
 */
public class Utils {

  public static List<LabeledData> loadLibSVM(String path, int dim) throws IOException {
    List<LabeledData> list = new ArrayList<LabeledData>();
    BufferedReader reader = new BufferedReader(new FileReader(new File(path)));

    String line;
    int cnt = 0;
    while ((line = reader.readLine()) != null) {
      LabeledData labeledData = parseOneLineLibSVM(line, dim);
      list.add(labeledData);
      cnt ++;
      if (cnt % 1000000 == 0) {
        System.out.println("Finishing load " + cnt + " lines.");
      }
    }
    return list;
  }

  public static LabeledData parseOneLineLibSVM(String line, int dim) {
    String[] parts = line.split(" ");
    int label = Integer.parseInt(parts[0]);
    if (label == 0)
      label = -1;
    int length = parts.length - 1;
    int[] indices = new int[length];
//    double[] values = new double[length];
    for (int i = 0; i < length; i ++) {
      String kv = parts[i + 1];
      String[] kvParts = kv.split(":");
      int idx = Integer.parseInt(kvParts[0]);
//      double value = Double.parseDouble(kvParts[1]);
      indices[i] = idx;
//      values[i]  = value;
    }

    SparseVector data = new SparseVector(dim, indices);
    return new LabeledData(data, label);
  }

  public static int[] generateRandomPermutation(int size) {
    int[] array = new int[size];
    for (int i = 0; i < size; i ++)
      array[i] = i;
    IntArrays.shuffle(array, new Random(System.currentTimeMillis()));
    return array;
  }


  public static DenseMap[] LoadLibSVMByFeature(String path, int featureDim,
                                               int sampleDim, double trainRatio) throws IOException{
    //Feature and Label(dimension: featureDim+1)
    DenseMap [] features = new DenseMap[featureDim + 1];
    for(int i = 0; i <= featureDim; i++){
      features[i] = new DenseMap();
    }
    BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
    String line;
    int cnt = 0;
    while ((line = reader.readLine()) != null && cnt < trainRatio * sampleDim) {
      String []parts = line.split(" ");
      //Label(y)
      features[featureDim].add(cnt, Double.parseDouble(parts[0]));
      for(int i = 0; i < parts.length - 1; i++){
        String kv = parts[i + 1];
        String[] kvParts = kv.split(":");
        int idx = Integer.parseInt(kvParts[0]);
//       double value = Double.parseDouble(kvParts[1]);
        features[idx].add(cnt, 1);
      }
      cnt++;
    }
    return features;
  }

  public static double soft_threshold(double th, double value){
    if(value > th){
      return value - th;
    }else if (value < -th){
      return value + th;
    }else{
      return 0;
    }
  }
}
