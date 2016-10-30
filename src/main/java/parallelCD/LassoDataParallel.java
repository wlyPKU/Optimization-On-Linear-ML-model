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
import model.Lasso;
/**
 * Created by WLY on 2016/9/4.
 */
//  数据并行
public class LassoDataParallel extends Lasso{
    private static long start;
    private static DenseVector model;
    private static DenseVector localModel[];
    private static double featureSquare[];
    private static double residual[];
    private static SparseMap[][] features;
    private static double lambda;
    private static double trainRatio = 0.5;
    private static int threadNum;
    private static int featureDimension;
    private static int sampleDimension;

    public class executeRunnable implements Runnable
    {
        private int threadID;
        public executeRunnable(int threadID){
            this.threadID = threadID;
        }
        public void run() {
            for(int j = 0; j < featureDimension; j++){
                double oldValue = localModel[threadID].values[j];
                double updateValue = 0;
                //No need for the calculation:
                if(featureSquare[j] == 0){
                    continue;
                }
                ObjectIterator<Int2DoubleMap.Entry> iter =  features[threadID][j].map.int2DoubleEntrySet().iterator();
                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    double xj = entry.getDoubleValue();
                    updateValue += xj * residual[idx];
                }
                updateValue /= featureSquare[j];
                localModel[threadID].values[j] = Utils.soft_threshold(lambda / featureSquare[j],
                        localModel[threadID].values[j] + updateValue);

                iter =  features[threadID][j].map.int2DoubleEntrySet().iterator();

                while (iter.hasNext()) {
                    Int2DoubleMap.Entry entry = iter.next();
                    int idx = entry.getIntKey();
                    double value = entry.getDoubleValue();
                    residual[idx] -= (localModel[threadID].values[j] - oldValue) * value;
                }
            }
            model.plusDense(localModel[threadID]);
        }
    }


    private void trainCore(List<LabeledData> labeledData) {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);

        featureSquare = new double[featureDimension];
        residual = new double[sampleDimension];
        for (int j = 0; j < features[0].length - 1; j++) {
            featureSquare[j] = 0;
            for(int i = 0; i < threadNum; i++) {
                for (Double v : features[i][j].map.values())
                    featureSquare[j] += v * v;
            }
        }

        for(int i = 0; i < threadNum; i++) {
            ObjectIterator<Int2DoubleMap.Entry> iter = features[i][featureDimension].map.int2DoubleEntrySet().iterator();
            while (iter.hasNext()) {
                Int2DoubleMap.Entry entry = iter.next();
                int idx = entry.getIntKey();
                double y = entry.getDoubleValue();
                residual[idx] = y;
            }
        }
        DenseVector oldModel = new DenseVector(featureDimension);

        long totalBegin = System.currentTimeMillis();

        for (int i = 0; ; i ++) {
            long startTrain = System.currentTimeMillis();

            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            Arrays.fill(model.values, 0);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                threadPool.execute(new executeRunnable(threadID));
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
            model.allDividedBy(threadNum);

            long trainTime = System.currentTimeMillis() - startTrain;
            System.out.println("trainTime " + trainTime + " ");
            testAndSummary(trainCorpus, testCorpus, model, lambda);

            System.out.println("totaltime " + (System.currentTimeMillis() - totalBegin) );
            if(converge(oldModel, model)){
                if(earlyStop)
                    break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, featureDimension);

            for(int idx = 0; idx < threadNum; idx ++){
                System.arraycopy(model.values, 0, localModel[idx].values, 0, featureDimension);
            }
            for(int idx = 0; idx < trainCorpus.size(); idx++){
                residual[idx] = labeledData.get(idx).label
                        - model.dot(labeledData.get(idx).data);
            }
            long nowCost = System.currentTimeMillis() - start;
            if(nowCost > maxTimeLimit){
                break;
                //break;
            }
        }
    }

    public static void train(List<LabeledData> labeledData) {
        LassoDataParallel lassoCD = new LassoDataParallel();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.

        model = new DenseVector(featureDimension);
        localModel = new DenseVector[threadNum];
        for(int i = 0; i < threadNum; i++){
            localModel[i] = new DenseVector(featureDimension);
        }
        Arrays.fill(model.values, 0);

        start = System.currentTimeMillis();
        lassoCD.trainCore(labeledData);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelCD.LassoDataParallel threadNum FeatureDim SampleDim train_path lambda trainRatio");
        threadNum=Integer.parseInt(argv[0]);
        featureDimension = Integer.parseInt(argv[1]);
        sampleDimension = Integer.parseInt(argv[2]);
        String path = argv[3];
        lambda = Double.parseDouble(argv[4]);
        trainRatio = 0.5;
        for(int i = 0; i < argv.length - 1; i++){
            if(argv[i].equals("EarlyStop")){
                earlyStop = Boolean.parseBoolean(argv[i + 1]);
            }
            if(argv[i].equals("TimeLimit")){
                maxTimeLimit = Double.parseDouble(argv[i + 1]);
            }
            if(argv[i].equals("StopDelta")){
                stopDelta = Double.parseDouble(argv[i + 1]);
            }
            if(argv[i].equals("TrainRatio")){
                trainRatio = Double.parseDouble(argv[i+1]);
                if(trainRatio >= 1 || trainRatio <= 0){
                    System.out.println("Error Train Ratio!");
                    System.exit(1);
                }            }
        }
        System.out.println("ThreadNum " + threadNum);
        System.out.println("StopDelta " + stopDelta);
        System.out.println("FeatureDimension " + featureDimension);
        System.out.println("SampleDimension " + sampleDimension);
        System.out.println("File Path " + path);
        System.out.println("Lambda " + lambda);
        System.out.println("TrainRatio " + trainRatio);
        System.out.println("TimeLimit " + maxTimeLimit);
        System.out.println("EarlyStop " + earlyStop);
        long startLoad = System.currentTimeMillis();
        features = Utils.LoadLibSVMByFeatureSplit(path, featureDimension, sampleDimension, trainRatio, threadNum);
        List<LabeledData> labeledData = Utils.loadLibSVM(path, featureDimension);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        train(labeledData);
    }
}