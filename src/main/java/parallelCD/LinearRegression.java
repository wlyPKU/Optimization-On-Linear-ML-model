package parallelCD;

import Utils.Utils;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import math.DenseVector;
import math.SparseMap;

import java.util.Arrays;
import java.util.List;
import Utils.LabeledData;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by WLY on 2016/9/4.
 */
public class LinearRegression extends model.LinearRegression{

    static double residual[];
    static DenseVector model;
    static double featureSquare[];
    static SparseMap[] features;
    static double trainRatio = 0.5;

    public class executeRunnable implements Runnable
    {
        int from, to;
        public executeRunnable(int from, int to){
            this.from = from;
            this.to = to;

        }
        public void run() {
            localTrain();
        }
        public void localTrain() {
            for(int j = from; j < to; j++){
                double oldValue = model.values[j];
                double updateValue = 0;

                ObjectIterator<Int2DoubleMap.Entry> iter =  features[j].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    double xj = entry.getDoubleValue();
                    updateValue += xj * residual[idx];
                }
                updateValue /= featureSquare[j];
                model.values[j] += updateValue;

                iter =  features[j].map.int2DoubleEntrySet().iterator();

                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    double value = entry.getDoubleValue();
                    residual[idx] -= (model.values[j] - oldValue) * value;
                }
            }
        }
    }


    public void trainCore(List<LabeledData> labeledData, int threadNum) {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        int featureDim = features.length - 1;
        int sampleSize = features[featureDim].map.size();
        featureSquare = new double[featureDim];
        residual = new double[sampleSize];
        for(int i = 0; i < featureDim; i++){
            featureSquare[i] = 0;
            for(Double v: features[i].map.values()){
                featureSquare[i] += v * v;
            }
            if(featureSquare[i] == 0){
                featureSquare[i] = Double.MAX_VALUE;
            }
        }
        ObjectIterator<Int2DoubleMap.Entry> iter =  features[featureDim].map.int2DoubleEntrySet().iterator();
        while (iter.hasNext()) {
            Int2DoubleMap.Entry entry = iter.next();
            int idx = entry.getIntKey();
            double y = entry.getDoubleValue();
            residual[idx] = y;
        }
        DenseVector oldModel = new DenseVector(featureDim);
        for (int i = 0; i < 100; i ++) {
            long startTrain = System.currentTimeMillis();
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                int from = featureDim * threadID / threadNum;
                int to = featureDim * (threadID + 1) / threadNum;
                threadPool.execute(new executeRunnable(from, to));
            }
            threadPool.shutdown();
            while (!threadPool.isTerminated()) {
                try {
                    threadPool.awaitTermination(1, TimeUnit.MILLISECONDS);
                } catch (InterruptedException e) {
                    System.out.println("Waiting.");
                    e.printStackTrace();
                }
            }

            long trainTime = System.currentTimeMillis() - startTrain;
            long startTest = System.currentTimeMillis();

            double loss = test(trainCorpus, model);
            double accuracy = test(testCorpus, model);
            long testTime = System.currentTimeMillis() - startTest;
            System.out.println("loss=" + loss + " testResidual=" + accuracy +
                    " trainTime=" + trainTime + " testTime=" + testTime);
            double []trainAccuracy = Utils.LinearAccuracy(trainCorpus, model);
            double []testAccuracy = Utils.LinearAccuracy(testCorpus, model);
            System.out.println("Train Accuracy:");
            Utils.printAccuracy(trainAccuracy);
            System.out.println("Test Accuracy:");
            Utils.printAccuracy(testAccuracy);

            if(converage(oldModel, model)){
                //break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, featureDim);
        }
    }

    public static void train(List<LabeledData> labeledData, int threadNum) {
        int dimension = features.length;
        LinearRegression linearCD = new LinearRegression();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        model = new DenseVector(dimension);
        Arrays.fill(model.values, 0);
        long start = System.currentTimeMillis();
        linearCD.trainCore(labeledData, threadNum);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelCD.LinearRegression threadNum FeatureDim SampleDim train_path trainRatio");
        int threadNum=Integer.parseInt(argv[0]);
        int featureDim = Integer.parseInt(argv[1]);
        int sampleDim = Integer.parseInt(argv[2]);
        String path = argv[3];
        trainRatio = 0.5;
        if(argv.length >= 5){
            trainRatio = Double.parseDouble(argv[4]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }
        long startLoad = System.currentTimeMillis();
        features = Utils.LoadLibSVMByFeature(path, featureDim, sampleDim, trainRatio);
        List<LabeledData> labeledData = Utils.loadLibSVM(path, featureDim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        train(labeledData, threadNum);
    }
}