package parallelGD;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by WLY on 2016/9/3.
 */
public class LogisticRegressionPS extends model.LogisticRegression{
    public static long start;

    public DenseVector globalModelOfU;
    public DenseVector globalModelOfV;
    public static double trainRatio = 0.5;
    public static double lambda = 0.1;
    public static int threadNum;

    public static double learningRate = 0.001;
    public int iteration = 0;


    public void setNewLearningRate(){
    }

    public class executeRunnable implements Runnable
    {
        List<LabeledData> localList;
        double lambda;
        int globalCorpusSize;
        int threadID;
        public executeRunnable(int threadID,List<LabeledData> list, double lambda, int globalCorpusSize){
            this.threadID = threadID;
            localList = list;
            this.lambda = lambda;
            this.globalCorpusSize = globalCorpusSize;

        }
        public void run() {
            sgdOneEpoch(localList, learningRate, lambda);
        }
        public void sgdOneEpoch(List<LabeledData> list, double lr, double lambda) {
            double modelPenalty = -lr * lambda / globalCorpusSize;
            for (LabeledData labeledData: list) {
                double predictValue = globalModelOfU.dot(labeledData.data) - globalModelOfV.dot(labeledData.data);
                double tmpValue = 1.0 / (1.0 + Math.exp(labeledData.label * predictValue));
                double scala = tmpValue * labeledData.label;
                globalModelOfU.plusSparse(labeledData.data, modelPenalty);
                globalModelOfU.plusGradient(labeledData.data, scala * lr);
                globalModelOfU.positiveOrZero(labeledData.data);

                predictValue = globalModelOfU.dot(labeledData.data) - globalModelOfV.dot(labeledData.data);
                tmpValue = 1.0 / (1.0 + Math.exp(labeledData.label * predictValue));
                scala = tmpValue * labeledData.label;
                globalModelOfV.plusSparse(labeledData.data, modelPenalty);
                globalModelOfV.plusGradient(labeledData.data, - scala * lr);
                globalModelOfV.positiveOrZero(labeledData.data);
            }
        }
    }

    public void train(List<LabeledData> corpus, DenseVector modelOfU, DenseVector modelOfV) {
        Collections.shuffle(corpus);
        int size = corpus.size();
        int end = (int) (size * trainRatio);
        List<LabeledData> trainCorpus = corpus.subList(0, end);
        List<LabeledData> testCorpus = corpus.subList(end, size);
        List<List<LabeledData>> ThreadTrainCorpus = new ArrayList<List<LabeledData>>();
        for(int threadID = 0; threadID < threadNum; threadID++){
            int from = end * threadID / threadNum;
            int to = end * (threadID + 1) / threadNum;
            List<LabeledData> threadCorpus = corpus.subList(from, to);
            ThreadTrainCorpus.add(threadCorpus);
        }

        globalModelOfU = new DenseVector(modelOfU.dim);
        globalModelOfV = new DenseVector(modelOfV.dim);

        DenseVector model = new DenseVector(modelOfU.dim);
        DenseVector oldModel = new DenseVector(model.dim);

        long totalBegin = System.currentTimeMillis();

        long totalIterationTime = 0;
        for (int i = 0; ; i ++) {
            long startTrain = System.currentTimeMillis();
            //TODO StepSize tuning:  c/k(k=0,1,2...) or backtracking line search
            System.out.println("learning rate " + learningRate);

            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                threadPool.execute(new executeRunnable(threadID, ThreadTrainCorpus.get(threadID), lambda, corpus.size()));
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
            System.arraycopy(globalModelOfU.values, 0, modelOfU.values, 0, modelOfU.dim);
            System.arraycopy(globalModelOfV.values, 0, modelOfV.values, 0, modelOfV.dim);
            for(int j = 0; j < model.dim; j++){
                model.values[j] = globalModelOfU.values[j] - globalModelOfV.values[j];
            }
            long trainTime = System.currentTimeMillis() - startTrain;
            System.out.println("Iteration " + i);

            System.out.println("trainTime " + trainTime + " ");
            totalIterationTime += trainTime;
            System.out.println("totalIterationTime " + totalIterationTime);
            testAndSummary(trainCorpus, testCorpus, model, lambda);
            System.out.println("totaltime " + (System.currentTimeMillis() - totalBegin) );
            iteration++;
            setNewLearningRate();
            if(modelType == 1) {
                if (totalIterationTime > maxTimeLimit) {
                    break;
                }
            }else if(modelType == 0){
                if(i > maxIteration){
                    break;
                }
            }else if (modelType == 2){
                if(converge(oldModel, model)){
                    break;
                }
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, oldModel.values.length);

        }
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelGD.LogisticRegression threadID FeatureDim train_path lambda learningRate [trainRatio]");
        threadNum = Integer.parseInt(argv[0]);
        int dimension = Integer.parseInt(argv[1]);
        String path = argv[2];
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dimension);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        lambda = Double.parseDouble(argv[3]);
        learningRate = Double.parseDouble(argv[4]);
        for(int i = 0; i < argv.length - 1; i++){
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
                }
            }
            if(argv[i].equals("MaxIteration")){
                maxIteration = Integer.parseInt(argv[i + 1]);
            }
            if(argv[i].equals("Model")){
                //0: maxIteration  1: maxTime 2: earlyStop
                modelType = Integer.parseInt(argv[i + 1]);
            }
        }
        System.out.println("ThreadNum " + threadNum);
        System.out.println("StopDelta " + stopDelta);
        System.out.println("FeatureDimension " + dimension);
        System.out.println("LearningRate " + learningRate);
        System.out.println("File Path " + path);
        System.out.println("Lambda " + lambda);
        System.out.println("TrainRatio " + trainRatio);
        System.out.println("TimeLimit " + maxTimeLimit);
        System.out.println("ModelType " + modelType);
        System.out.println("Iteration Limit " + maxIteration);
        LogisticRegressionPS lr = new LogisticRegressionPS();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        DenseVector modelOfU = new DenseVector(dimension);
        DenseVector modelOfV = new DenseVector(dimension);
        start = System.currentTimeMillis();
        lr.train(corpus, modelOfU, modelOfV);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
}
