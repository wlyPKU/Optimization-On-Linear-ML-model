package parallelCD;

import Utils.LabeledData;
import Utils.Utils;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import math.DenseVector;
import math.SparseMap;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by WLY on 2016/9/4.
 */
// Block Coordinate Descent
//  model       每个线程独立(虽然更改同样一份model,但因为实际计算中使用的是residual,因此每个线程独立)
//  residual    每个线程独立
public class LassoBCD extends model.Lasso{
    private static long start;
    private static double residual[][];
    private static DenseVector model;
    private static double featureSquare[];
    private static SparseMap[] features;
    private static double lambda;
    private static double trainRatio = 0.5;
    private static int threadNum;
    private static int featureDimension;
    private static int sampleDimension;

    public class executeRunnable implements Runnable
    {
        int from, to;
        int threadID;
        public executeRunnable(int threadID, int from, int to){
            this.from = from;
            this.to = to;
            this.threadID = threadID;
        }
        public void run() {
            for(int j = from; j < to; j++){
                double oldValue = model.values[j];
                double updateValue = 0;

                ObjectIterator<Int2DoubleMap.Entry> iter =  features[j].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    double xj = entry.getDoubleValue();
                    updateValue += xj * residual[threadID][idx];
                }
                updateValue /= featureSquare[j];
                model.values[j] += updateValue;
                model.values[j] = Utils.soft_threshold(lambda / featureSquare[j], model.values[j]);

                iter =  features[j].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    double value = entry.getDoubleValue();
                    residual[threadID][idx] -= (model.values[j] - oldValue) * value;
                }
            }
        }
    }


    private void trainCore(List<LabeledData> labeledData) {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        featureSquare = new double[featureDimension];
        residual = new double[threadNum][sampleDimension];
        for(int i = 0; i < featureDimension; i++){
            featureSquare[i] = 0;
            for(Double v: features[i].map.values()){
                featureSquare[i] += v * v;
            }
        }

        ObjectIterator<Int2DoubleMap.Entry> iter =  features[featureDimension].map.int2DoubleEntrySet().iterator();
        while (iter.hasNext()) {
            Int2DoubleMap.Entry entry = iter.next();
            int idx = entry.getIntKey();
            double y = entry.getDoubleValue();
            for(int j = 0; j < threadNum; j++){
                residual[j][idx] = y;
            }
        }
        DenseVector oldModel = new DenseVector(featureDimension);

        long totalBegin = System.currentTimeMillis();


        for (int i = 0; ; i ++) {
            iter =  features[featureDimension].map.int2DoubleEntrySet().iterator();
            while (iter.hasNext()) {
                Int2DoubleMap.Entry entry = iter.next();
                int idx = entry.getIntKey();
                double y = entry.getDoubleValue();
                for(int j = 0; j < threadNum; j++){
                    residual[j][idx] = y - model.dot(trainCorpus.get(idx).data);
                }
            }
            long startTrain = System.currentTimeMillis();
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                int from = featureDimension * threadID / threadNum;
                int to = featureDimension * (threadID + 1) / threadNum;
                threadPool.execute(new executeRunnable(threadID, from, to));
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
            System.out.println("trainTime " + trainTime + " ");
            testAndSummary(trainCorpus, testCorpus, model, lambda);
            System.out.println("totaltime " + (System.currentTimeMillis() - totalBegin) );
            if(converge(oldModel, model)){
                break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, featureDimension);

            long nowCost = System.currentTimeMillis() - start;
            if(nowCost > 60000){
                break;
                //break;
            }
        }
    }

    public static void train(List<LabeledData> labeledData) {
        LassoBCD lassoModelParallelCD = new LassoBCD();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        model = new DenseVector(featureDimension);
        Arrays.fill(model.values, 0);
        start = System.currentTimeMillis();
        lassoModelParallelCD.trainCore(labeledData);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelCD.LassoBCD threadNum FeatureDim SampleDim train_path lambda trainRatio");
        threadNum=Integer.parseInt(argv[0]);
        featureDimension = Integer.parseInt(argv[1]);
        sampleDimension = Integer.parseInt(argv[2]);
        String path = argv[3];
        lambda = Double.parseDouble(argv[4]);
        trainRatio = 0.5;
        if(argv.length >= 6){
            trainRatio = Double.parseDouble(argv[5]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }
        long startLoad = System.currentTimeMillis();
        features = Utils.LoadLibSVMByFeature(path, featureDimension, sampleDimension, trainRatio);
        List<LabeledData> labeledData = Utils.loadLibSVM(path, featureDimension);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        train(labeledData);
    }
}