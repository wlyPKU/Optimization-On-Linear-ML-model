package CoordinateDescent;

import it.unimi.dsi.fastutil.doubles.DoubleComparator;
import math.SparseMap;
import math.DenseVector;
import Utils.LabeledData;
import Utils.Utils;
import Utils.Sort;
import java.util.List;
import java.util.Map;

/**
 * Created by 王羚宇 on 2016/7/26.
 */
//Ref: http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf Pg. 7,14,18
public class LogisticRegressionSCD {
    private double logLoss(List<LabeledData> list, DenseVector model, double lambda) {
        double loss = 0.0;
        for (LabeledData labeledData: list) {
            double p = model.dot(labeledData.data);
            double z = p * labeledData.label;
            if (z > 18) {
                loss += Math.exp(-z);
            } else if (z < -18) {
                loss += -z;
            } else {
                loss += Math.log(1 + Math.exp(-z));
            }
        }
        for(Double v : model.values){
            loss += lambda * (v > 0? v : -v);
        }
        return loss;
    }
    @SuppressWarnings("unused")
    private double lossFunctionValue(List<LabeledData> labeledData,
                                     DenseVector model, double lambda){
        double result = 0;
        for(LabeledData l: labeledData){
            double exp = Math.exp(- l.label * model.dot(l.data));
            double loss = Math.log(1 + exp);
            result += loss;
        }
        for(Double w : model.values){
            result += lambda * Math.abs(w);
        }
        return result;
    }
    @SuppressWarnings("unused")
    private double getVectorLength(DenseVector model){
        double length = 0;
        for(Double w: model.values){
            length += w * w;
        }
        return Math.sqrt(length);
    }
    public void train(SparseMap[] features, List<LabeledData> labeledData,
                      DenseVector modelOfU, DenseVector modelOfV, double lambda, double trainRatio) {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        int featureDim = features.length - 1;

        double predictValue[] = new double[labeledData.size()];

        DenseVector model = new DenseVector(featureDim);
        for (int i = 0; i < 30; i ++) {
            for(int idx = 0; idx < labeledData.size(); idx++){
                LabeledData l = labeledData.get(idx);
                predictValue[idx] = modelOfU.dot(l.data) - modelOfV.dot(l.data);
            }
            long startTrain = System.currentTimeMillis();
            //Update w+
            for(int fIdx = 0; fIdx < featureDim; fIdx++){
                //First Order L:
                double firstOrderL = 0;
                double oldValue = modelOfU.values[fIdx];
                for(Map.Entry<Integer, Double> m: features[fIdx].map.entrySet()){
                    int idx = m.getKey();
                    LabeledData l = labeledData.get(idx);
                    double tao = 1 / (1 + Math.exp( -l.label * predictValue[idx]));
                    firstOrderL += l.label * m.getValue() * (tao - 1);
                }
                /*
                for(int j = 0; j < features[fIdx].index.size(); j++){
                    int idx = features[fIdx].index.get(j);
                    LabeledData l = labeledData.get(idx);
                    double tao = 1 / (1 + Math.exp( -l.label * predictValue[idx]));
                    firstOrderL += l.label * features[fIdx].value.get(j) * (tao - 1);
                }
                */
                double Uj = 0.25 * 1 / lambda * trainCorpus.size();
                double updateValue = (1 + firstOrderL) / Uj;
                if(updateValue > modelOfU.values[fIdx]){
                    modelOfU.values[fIdx] = 0;
                }else{
                    modelOfU.values[fIdx] -= updateValue;
                }
                //Update predictValue
                for(Map.Entry<Integer, Double> m: features[fIdx].map.entrySet()){
                    int idx = m.getKey();
                    predictValue[idx] += m.getValue() * (modelOfU.values[fIdx] - oldValue);
                }
                /*
                for(int j = 0; j < features[fIdx].index.size(); j++){
                    int idx = features[fIdx].index.get(j);
                    predictValue[idx] += features[fIdx].value.get(j) * (modelOfU.values[fIdx] - oldValue);
                }
                */
            }
            //Update w-
            for(int fIdx = 0; fIdx < featureDim; fIdx++){
                //First Order L:
                double firstOrderL = 0;
                double oldValue = modelOfV.values[fIdx];
                for(Map.Entry<Integer, Double> m: features[fIdx].map.entrySet()){
                    int idx = m.getKey();
                    LabeledData l = labeledData.get(idx);
                    double tao = 1 / (1 + Math.exp( -l.label * predictValue[idx]));
                    firstOrderL += l.label * m.getValue() * (tao - 1);
                }

                double Uj = 0.25 * 1 / lambda * trainCorpus.size();
                double updateValue = (1 - firstOrderL) / Uj;
                if(updateValue > modelOfU.values[fIdx]){
                    modelOfV.values[fIdx] = 0;
                }else{
                    modelOfV.values[fIdx] -= updateValue;
                }
                for(Map.Entry<Integer, Double> m: features[fIdx].map.entrySet()){
                    int idx = m.getKey();
                    predictValue[idx] -= m.getValue() * (modelOfV.values[fIdx] - oldValue);
                }

            }
            for(int fIdx = 0; fIdx < featureDim; fIdx ++){
                model.values[fIdx] = modelOfU.values[fIdx] - modelOfV.values[fIdx];
            }
            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();

            double loss = logLoss(trainCorpus, model, lambda);

            double trainAuc = auc(trainCorpus, model);
            double testAuc = auc(testCorpus, model);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " trainAuc=" + trainAuc + " testAuc=" + testAuc +
                    " trainTime=" + trainTime + " testTime=" + testTime);
        }
    }


    public static void train(SparseMap[] corpus, List<LabeledData> labeledData,
                             double lambda, double trainRatio) {
        int dim = corpus.length;
        LogisticRegressionSCD lrSCD = new LogisticRegressionSCD();
        //http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf 3197-3200+
        DenseVector modelOfU = new DenseVector(dim);
        DenseVector modelOfV = new DenseVector(dim);
        for(int i = 0; i < dim; i++){
            modelOfU.values[i] = 0;
            modelOfV.values[i] = 0;
        }
        long start = System.currentTimeMillis();
        lrSCD.train(corpus, labeledData, modelOfU, modelOfV, lambda, trainRatio);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }

    private double auc(List<LabeledData> list, DenseVector model) {
        int length = list.size();
        System.out.println(length);
        double[] scores = new double[length];
        double[] labels = new double[length];

        int cnt = 0;
        for (LabeledData labeledData: list) {
            double z = model.dot(labeledData.data);
            double score = 1.0 / (1.0 + Math.exp(-z));

            scores[cnt] = score;
            labels[cnt] = labeledData.label;
            cnt ++;
        }

        Sort.quickSort(scores, labels, 0, length, new DoubleComparator() {

            public int compare(double i, double i1) {
                if (Math.abs(i - i1) < 10e-12) {
                    return 0;
                } else {
                    return i - i1 > 10e-12 ? 1 : -1;
                }
            }

            public int compare(Double o1, Double o2) {
                if (Math.abs(o1 - o2) < 10e-12) {
                    return 0;
                } else {
                    return o1 - o2 > 10e-12 ? 1 : -1;
                }
            }
        });

        long M = 0, N = 0;
        for (int i = 0; i < scores.length; i ++) {
            if (labels[i] == 1.0)
                M ++;
            else
                N ++;
        }

        double sigma = 0.0;
        for (long i = M + N - 1; i >= 0; i --) {
            if (labels[(int) i] == 1.0) {
                sigma += i;
            }
        }

        double auc = (sigma - (M + 1) * M / 2) / (M * N);
        System.out.println("sigma=" + sigma + " M=" + M + " N=" + N);
        return auc;
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: CoordinateDescent.LogisticRegressionSCD FeatureDim SampleDim train_path lamda trainRatio");
        int featureDim = Integer.parseInt(argv[0]);
        int sampleDim = Integer.parseInt(argv[1]);
        String path = argv[2];
        double lambda = Double.parseDouble(argv[3]);
        if(lambda <= 0){
            System.out.println("Please input a correct lambda (>0)");
            System.exit(2);
        }
        double trainRatio = 0.5;
        if(argv.length >= 5){
            trainRatio = Double.parseDouble(argv[4]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }
        long startLoad = System.currentTimeMillis();
        SparseMap[] features = Utils.LoadLibSVMByFeature(path, featureDim, sampleDim, trainRatio);
        List<LabeledData> labeledData = Utils.loadLibSVM(path, featureDim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        //TODO Need to think how to min hash numeric variables
        train(features, labeledData, lambda, trainRatio);
    }
}