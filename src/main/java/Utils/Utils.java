package Utils;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.IntArrays;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import math.DenseVector;
import math.SparseMap;
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
      if (cnt % 100000 == 0) {
        System.out.println("Finishing load " + cnt + " lines.");
      }
    }
    return list;
  }

  public static List<LabeledData> normalizeData(List<LabeledData> labeledData, int featureNum){
    double minFeatureValue[] = new double[featureNum];
    double maxFeatureValue[] = new double[featureNum];
    double sumFeatureValue[] = new double[featureNum];
    int nonZeroFeatures[] = new int[featureNum];
    Arrays.fill(minFeatureValue, Double.MAX_VALUE);
    Arrays.fill(maxFeatureValue, - Double.MAX_VALUE);
    Arrays.fill(nonZeroFeatures, 0);
    Arrays.fill(sumFeatureValue, 0);
    for(LabeledData l: labeledData){
      if(l.data.values != null){
        for(int i = 0; i < l.data.indices.length; i++){
            minFeatureValue[l.data.indices[i]] = Math.min(minFeatureValue[l.data.indices[i]], l.data.values[i]);
            maxFeatureValue[l.data.indices[i]] = Math.max(maxFeatureValue[l.data.indices[i]], l.data.values[i]);
            nonZeroFeatures[l.data.indices[i]] ++;
            sumFeatureValue[l.data.indices[i]] += l.data.values[i];
        }
      }
    }
    for(LabeledData l: labeledData){
      for(int i = 0; i < l.data.indices.length; i++){
        /*
        if(l.data.values != null && minFeatureValue[l.data.indices[i]] != maxFeatureValue[l.data.indices[i]]
                && nonZeroFeatures[l.data.indices[i]]!= 0 ){
          l.data.values[i] = (l.data.values[i] - sumFeatureValue[l.data.indices[i]] / nonZeroFeatures[l.data.indices[i]]) /
                  (maxFeatureValue[l.data.indices[i]] - minFeatureValue[l.data.indices[i]]);
        }
        */
        if(l.data.values != null && maxFeatureValue[l.data.indices[i]] != 0 &&nonZeroFeatures[l.data.indices[i]]!= 0 ){
          l.data.values[i] /= Math.abs(maxFeatureValue[l.data.indices[i]]);
        }
      }
    }
    return labeledData;
  }
  public static Vector<LabeledData> loadLibSVMVector(String path, int dim) throws IOException {
    Vector<LabeledData> list = new Vector<LabeledData>();
    BufferedReader reader = new BufferedReader(new FileReader(new File(path)));

    String line;
    int cnt = 0;
    while ((line = reader.readLine()) != null) {
      LabeledData labeledData = parseOneLineLibSVM(line, dim);
      list.add(labeledData);
      cnt ++;
      if (cnt % 100000 == 0) {
        System.out.println("Finishing load " + cnt + " lines.");
      }
    }
    return list;
  }
  public static LabeledData[] loadLibSVMArray(String path, int dim, boolean doShuffle) throws IOException {
    Vector<LabeledData> list = new Vector<LabeledData>();
    BufferedReader reader = new BufferedReader(new FileReader(new File(path)));

    String line;
    int cnt = 0;
    while ((line = reader.readLine()) != null) {
      LabeledData labeledData = parseOneLineLibSVM(line, dim);
      list.add(labeledData);
      cnt ++;
      if (cnt % 100000 == 0) {
        System.out.println("Finishing load " + cnt + " lines.");
      }
    }
    if(doShuffle == true){
      Collections.shuffle(list);
    }
    LabeledData[] result = new LabeledData[list.size()];
    for(int i = 0; i < list.size(); i++){
      result[i] = new LabeledData(list.get(i).data, list.get(i).label);
    }
    return result;
  }

  public static List<LabeledData> loadLibSVMLoss0_1(String path, int dim) throws IOException {
        List<LabeledData> list = new ArrayList<LabeledData>();
        BufferedReader reader = new BufferedReader(new FileReader(new File(path)));

        String line;
        int cnt = 0;
        while ((line = reader.readLine()) != null) {
            LabeledData labeledData = parseOneLineLibSVMLoss0_1(line, dim);
            list.add(labeledData);
            cnt ++;
            if (cnt % 100000 == 0) {
                System.out.println("Finishing load " + cnt + " lines.");
            }
        }
        return list;
    }

  public static List<LabeledData> loadLibSVMWithBiase(String path, int dim) throws IOException {
    List<LabeledData> list = new ArrayList<LabeledData>();
    BufferedReader reader = new BufferedReader(new FileReader(new File(path)));

    String line;
    int cnt = 0;
    while ((line = reader.readLine()) != null) {
      LabeledData labeledData = parseOneLineLibSVMWithBiase(line, dim);
      list.add(labeledData);
      cnt ++;
      if (cnt % 100000 == 0) {
        System.out.println("Finishing load " + cnt + " lines.");
      }
    }
    return list;
  }

  //Divide the corpus into [threadNum] parts by features, the corpus' sample size is [size],
  // and the feature size is [dim].
  public static List<List<LabeledData>> loadLibSVMSplit(String path, int dim,
                                                  int threadNum, int size) throws IOException {

    List<List<LabeledData>> resultList = new ArrayList<List<LabeledData>>();
    for(int threadID = 0; threadID < threadNum; threadID++){
      int from = dim * threadID / threadNum;
      int to = dim * (threadID + 1) / threadNum;
      List<LabeledData> list = new ArrayList<LabeledData>();
      BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
      String line;
      int cnt = 0;
      while ((line = reader.readLine()) != null && size > cnt) {
        LabeledData labeledData = parseOneLineLibSVM(line, dim, from, to);
        list.add(labeledData);
        cnt ++;
      }
      resultList.add(list);
    }
    return resultList;
  }

  private static LabeledData parseOneLineLibSVM(String line, int dim, int from , int to) {
    String[] parts = line.split(" ");
    double label = Double.parseDouble(parts[0]);
    if (label == 0)
      label = -1;
    int length = parts.length - 1;
    int featureCount = 0;
    for (int i = 0; i < length; i ++) {
      String kv = parts[i + 1];
      String[] kvParts = kv.split(":");
      int idx = Integer.parseInt(kvParts[0]);
      if(idx >= from && idx < to){
        featureCount++;
      }
    }
    int[] indices = new int[featureCount];
    double[] values = new double[featureCount];
    int index = 0;
    for (int i = 0; i < length; i ++) {
      String kv = parts[i + 1];
      String[] kvParts = kv.split(":");
      int idx = Integer.parseInt(kvParts[0]);
      double value = Double.parseDouble(kvParts[1]);
      if(idx >= from && idx < to){
        indices[index] = idx;
        values[index]  = value;
        index++;
      }
    }

    SparseVector data = new SparseVector(dim, indices, values);
    return new LabeledData(data, label);
  }
  private static LabeledData parseOneLineLibSVMLoss0_1(String line, int dim) {
        String[] parts = line.split(" ");
        double label = Double.parseDouble(parts[0]);
        int length = parts.length - 1;
        int[] indices = new int[length];
        double[] values = new double[length];
        for (int i = 0; i < length; i ++) {
            String kv = parts[i + 1];
            String[] kvParts = kv.split(":");
            int idx = Integer.parseInt(kvParts[0]);
            double value = Double.parseDouble(kvParts[1]);
            indices[i] = idx;
            values[i]  = value;
        }
        SparseVector data = new SparseVector(dim, indices, values);
        return new LabeledData(data, label);
  }
  private static LabeledData parseOneLineLibSVM(String line, int dim) {
    String[] parts = line.split(" ");
    double label = Double.parseDouble(parts[0]);
    if (label == 0)
      label = -1;
    int length = parts.length - 1;
    int[] indices = new int[length];
    double[] values = new double[length];
    for (int i = 0; i < length; i ++) {
      String kv = parts[i + 1];
      String[] kvParts = kv.split(":");
      int idx = Integer.parseInt(kvParts[0]);
      double value = Double.parseDouble(kvParts[1]);
      indices[i] = idx;
      values[i]  = value;
    }
    SparseVector data = new SparseVector(dim, indices, values);
    return new LabeledData(data, label);
  }

    private static LabeledData parseOneLineLibSVMWithBiase(String line, int dim) {
        String[] parts = line.split(" ");
        double label = Double.parseDouble(parts[0]);
        if (label == 0)
            label = -1;
        int length = parts.length - 1;
        int[] indices = new int[length + 1];
        double[] values = new double[length + 1];
        for (int i = 0; i < length; i ++) {
            String kv = parts[i + 1];
            String[] kvParts = kv.split(":");
            int idx = Integer.parseInt(kvParts[0]);
            double value = Double.parseDouble(kvParts[1]);
            indices[i] = idx;
            values[i]  = value;
        }
        //Add biase
        indices[length] = dim - 1;
        values[length] = 1;
        SparseVector data = new SparseVector(dim, indices, values);
        return new LabeledData(data, label);
    }

  static int[] generateRandomPermutation(int size) {
    int[] array = new int[size];
    for (int i = 0; i < size; i ++)
      array[i] = i;
    IntArrays.shuffle(array, new Random(System.currentTimeMillis()));
    return array;
  }


  public static SparseVector[] generateSpareVector(SparseMap[] maps){
      SparseVector result[] = new SparseVector[maps.length];
      for(int i = 0; i < maps.length; i++){
        int indices[] = new int[maps[i].map.int2DoubleEntrySet().size()];
        double values[] = new double[maps[i].map.int2DoubleEntrySet().size()];
        ObjectIterator<Int2DoubleMap.Entry> iter =  maps[i].map.int2DoubleEntrySet().iterator();
        int j = 0;
        while (iter.hasNext()) {
          Int2DoubleMap.Entry entry = iter.next();
          int idx = entry.getIntKey();
          double value = entry.getDoubleValue();
          indices[j] = idx;
          values[j] = value;
          j++;
        }
        result[i] = new SparseVector(indices.length, indices, values);

      }
      return result;
  }
  @SuppressWarnings("unused")
  public static SparseMap[] LoadLibSVMByFeature(String path, int featureDim) throws IOException{
    //Feature and Label(dimension: featureDim+1)
    SparseMap[] features = new SparseMap[featureDim + 1];
    for(int i = 0; i <= featureDim; i++){
      features[i] = new SparseMap();
    }
    BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
    String line;
    int cnt = 0;
    while ((line = reader.readLine()) != null) {
      String []parts = line.split(" ");
      //Label(y)
      features[featureDim].add(cnt, Double.parseDouble(parts[0]));
      for(int i = 0; i < parts.length - 1; i++){
        String kv = parts[i + 1];
        String[] kvParts = kv.split(":");
        int idx = Integer.parseInt(kvParts[0]);
        double value = Double.parseDouble(kvParts[1]);
        features[idx].add(cnt, value);
      }
      cnt++;
    }
    return features;
  }

  public static SparseMap[] LoadLibSVMFromLabeledData(List<LabeledData> list, int featureDim, double trainRatio){
    //Feature and Label(dimension: featureDim+1)
    SparseMap[] features = new SparseMap[featureDim + 1];
    for(int i = 0; i <= featureDim; i++){
      features[i] = new SparseMap();
    }
    for(int i = 0; i < (int)(list.size() * trainRatio); i++){
      LabeledData tmp = list.get(i);
      for(int j = 0; j < tmp.data.indices.length; j++){
        if(tmp.data.values == null){
          features[tmp.data.indices[j]].add(i, 1);
        }else{
          features[tmp.data.indices[j]].add(i, tmp.data.values[j]);

        }
      }
      features[featureDim].add(i, tmp.label);
    }
    return features;
  }
  public static SparseMap[] LoadLibSVMFromLabeledData(LabeledData[] list, int featureDim, double trainRatio){
    //Feature and Label(dimension: featureDim+1)
    SparseMap[] features = new SparseMap[featureDim + 1];
    for(int i = 0; i <= featureDim; i++){
      features[i] = new SparseMap();
    }
    for(int i = 0; i < (int)(list.length * trainRatio); i++){
      LabeledData tmp = list[i];
      for(int j = 0; j < tmp.data.indices.length; j++){
        if(tmp.data.values == null){
          features[tmp.data.indices[j]].add(i, 1);
        }else{
          features[tmp.data.indices[j]].add(i, tmp.data.values[j]);

        }
      }
      features[featureDim].add(i, tmp.label);
    }
    return features;
  }
  public static SparseMap[] LoadLibSVMFromLabeledDataBySplit(List<LabeledData> list, int featureDim, double trainRatio,
                                                             double threadNum, double threadID){
    //Feature and Label(dimension: featureDim+1)
    SparseMap[] features = new SparseMap[featureDim + 1];
    for(int i = 0; i <= featureDim; i++){
      features[i] = new SparseMap();
    }
    for(int i = (int)((int)(list.size() * trainRatio) * threadID / threadNum);
       i < (int)((int)(list.size() * trainRatio) * (threadID + 1)/ threadNum); i++){
      LabeledData tmp = list.get(i);
      for(int j = 0; j < tmp.data.indices.length; j++){
        if(tmp.data.values == null){
          features[tmp.data.indices[j]].add(i, 1);
        }else{
          features[tmp.data.indices[j]].add(i, tmp.data.values[j]);

        }
      }
      features[featureDim].add(i, tmp.label);
    }
    return features;
  }
  @SuppressWarnings("unused")
  public static SparseMap[] LoadLibSVMByFeature(String path, int featureDim, int sampleDim, double trainRatio) throws IOException{
    //Feature and Label(dimension: featureDim+1)
    SparseMap[] features = new SparseMap[featureDim + 1];
    for(int i = 0; i <= featureDim; i++){
      features[i] = new SparseMap();
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
       double value = Double.parseDouble(kvParts[1]);
        features[idx].add(cnt, value);
      }
      cnt++;
    }
    return features;
  }
  @SuppressWarnings("unused")
  public static SparseMap[][] LoadLibSVMByFeatureSplit(String path, int featureDim,
      int sampleDim, double trainRatio, int splitNum) throws IOException{
    //Feature and Label(dimension: featureDim+1)
    SparseMap[][]resultSet = new SparseMap[splitNum][featureDim + 1];
    for(int i = 0; i < splitNum; i++){
      for(int j = 0; j <= featureDim; j++){
        resultSet[i][j] = new SparseMap();
      }
    }

    BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
    String line;
    int cnt = 0;
    int perSplitSize = (int)(Math.ceil(trainRatio * sampleDim / splitNum));
    while ((line = reader.readLine()) != null && cnt < trainRatio * sampleDim) {
      String []parts = line.split(" ");
      //Label(y)
      int splitIndex = cnt / perSplitSize;
      resultSet[splitIndex][featureDim].add(cnt, Double.parseDouble(parts[0]));
      for(int i = 0; i < parts.length - 1; i++){
        String kv = parts[i + 1];
        String[] kvParts = kv.split(":");
        int idx = Integer.parseInt(kvParts[0]);
        double value = Double.parseDouble(kvParts[1]);
        resultSet[splitIndex][idx].add(cnt, value);
      }
      cnt++;
    }
    return resultSet;
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
  public static double[] LinearAccuracy(List<LabeledData> list, DenseVector model){
    double []table = new double[16];
    for(int i = 0; i < 16; i++){
      table[i] = 0;
    }
    for(LabeledData l : list){
      double predictValue = model.dot(l.data);
      double delta = Math.abs(predictValue - l.label);
      //delta [0] < 0.01
      //delta [1] < 0.02
      //delta [2] < 0.05
      //delta [3] < 0.1
      //delta [4] < 0.2
      //delta [5] < 0.5
      //delta [6] < 1
      //delta [7] < 2
      //delta [8] < 5
      //delta [9] < 10
      //delta [10] < 20
      //delta [11] < 50
      //delta [12] < 100
      //delta [13] < 200
      //delta [14] < 500
      if(delta < 0.01){
        table[0] ++;
      }else if(delta < 0.02){
        table[1] ++;
      }else if(delta < 0.05){
        table[2] ++;
      }else if(delta < 0.1){
        table[3] ++;
      }else if(delta < 0.2){
        table[4] ++;
      }else if(delta < 0.5){
        table[5] ++;
      }else if(delta < 1){
        table[6] ++;
      }else if(delta < 2){
        table[7] ++;
      }else if(delta < 5){
        table[8] ++;
      }else if(delta < 10){
        table[9] ++;
      }else if(delta < 20){
        table[10] ++;
      }else if(delta < 50){
        table[11] ++;
      }else if(delta < 100){
        table[12] ++;
      }else if(delta < 200){
        table[13] ++;
      }else if(delta < 500) {
        table[14]++;
      }else{
        table[15]++;
      }
    }
    for(int i = 0; i < 16; i++){
      table[i] /= list.size();
    }
    return table;
  }
  @SuppressWarnings("unused")
  public static void printAccuracy(double[] accuracy){
    System.out.println("-----Loss<0.01: " + accuracy[0] * 100 + "%");
    System.out.println("-----Loss<0.02: " + accuracy[1] * 100 + "%");
    System.out.println("-----Loss<0.05: " + accuracy[2] * 100 + "%");
    System.out.println("-----Loss<0.1: " + accuracy[3] * 100 + "%");
    System.out.println("-----Loss<0.2: " + accuracy[4] * 100 + "%");
    System.out.println("-----Loss<0.5: " + accuracy[5] * 100 + "%");
    System.out.println("-----Loss<1: " + accuracy[6] * 100 + "%");
    System.out.println("-----Loss<2: " + accuracy[7] * 100 + "%");
    System.out.println("-----Loss<5: " + accuracy[8] * 100 + "%");
    System.out.println("-----Loss<10: " + accuracy[9] * 100 + "%");
    System.out.println("-----Loss<20: " + accuracy[10] * 100 + "%");
    System.out.println("-----Loss<50: " + accuracy[11] * 100 + "%");
    System.out.println("-----Loss<100: " + accuracy[12] * 100 + "%");
    System.out.println("-----Loss<200: " + accuracy[13] * 100 + "%");
    System.out.println("-----Loss<500: " + accuracy[14] * 100 + "%");
    System.out.println("-----Loss>500: " + accuracy[15] * 100 + "%");
  }
}
