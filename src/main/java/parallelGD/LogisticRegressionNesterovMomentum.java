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
public class LogisticRegressionNesterovMomentum extends model.LogisticRegression{

    static long start;
    private DenseVector globalModelOfU;
    private DenseVector globalModelOfV;
    private static double trainRatio = 0.5;
    private static double lambda = 0.1;
    private static int threadNum;

    private double[][] momentumV;
    private double[][] momentumU;

    double gamma = 0.9;
    static double eta = 0.005;

    public class executeRunnable implements Runnable
    {
        List<LabeledData> localList;
        DenseVector localModelOfU;
        DenseVector localModelOfV;
        double lambda;
        int globalCorpusSize;
        int threadID;

        public executeRunnable(List<LabeledData> list, DenseVector modelOfU, DenseVector modelOfV, double lambda,
                               int globalCorpusSize, int threadID){
            this.threadID = threadID;
            localList = list;
            localModelOfU = new DenseVector(modelOfU.dim);
            localModelOfV = new DenseVector(modelOfV.dim);
            System.arraycopy(modelOfU.values, 0, localModelOfU.values, 0, modelOfU.dim);
            System.arraycopy(modelOfV.values, 0, localModelOfV.values, 0, modelOfU.dim);
            this.lambda = lambda;
            this.globalCorpusSize = globalCorpusSize;

        }
        public void run() {
            sgdOneEpoch(localList, localModelOfU, localModelOfV, lambda);
            globalModelOfU.plusDense(localModelOfU);
            globalModelOfV.plusDense(localModelOfV);
        }
        private void sgdOneEpoch(List<LabeledData> list, DenseVector modelOfU,
                                DenseVector modelOfV, double lambda) {
            double modelPenalty = - lambda / globalCorpusSize;
            for (LabeledData labeledData: list) {

                double predictValue = modelOfU.dotNesterovMomentum(labeledData.data, momentumU[threadID], gamma) - modelOfV.dot(labeledData.data);
                double tmpValue = 1.0 / (1.0 + Math.exp(labeledData.label * predictValue));
                double scala = tmpValue * labeledData.label;
                for(int i = 0; i < labeledData.data.indices.length; i++){
                    momentumU[threadID][labeledData.data.indices[i]] *= gamma;
                    if(labeledData.data.values != null){
                        momentumU[threadID][labeledData.data.indices[i]] += eta * (scala *  labeledData.data.values[i] + modelPenalty);
                    }else{
                        momentumU[threadID][labeledData.data.indices[i]] += eta * (scala + modelPenalty);
                    }
                    modelOfU.values[labeledData.data.indices[i]] += momentumU[threadID][labeledData.data.indices[i]];
                }
                
                predictValue = modelOfU.dot(labeledData.data) - modelOfV.dotNesterovMomentum(labeledData.data, momentumV[threadID], gamma);
                tmpValue = 1.0 / (1.0 + Math.exp(labeledData.label * predictValue));
                scala = tmpValue * labeledData.label;
                for(int i = 0; i < labeledData.data.indices.length; i++){
                    momentumV[threadID][labeledData.data.indices[i]] *= gamma;
                    if(labeledData.data.values != null){
                        momentumV[threadID][labeledData.data.indices[i]] += eta * (-scala *  labeledData.data.values[i] + modelPenalty);
                    }else{
                        momentumV[threadID][labeledData.data.indices[i]] += eta * (-scala + modelPenalty);
                    }
                    momentumV[threadID][labeledData.data.indices[i]] += eta * (-scala *  labeledData.data.values[i] + modelPenalty);
                    modelOfV.values[labeledData.data.indices[i]] += momentumV[threadID][labeledData.data.indices[i]];
                }
                modelOfV.positiveOrZero(labeledData.data);
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

        momentumV = new double[threadNum][modelOfU.dim];
        momentumU = new double[threadNum][modelOfU.dim];
        for(int i = 0; i < threadNum; i++){
            for(int j = 0; j < modelOfU.dim; j++){
                momentumU[i][j] = 0;
                momentumV[i][j] = 0;
            }
        }

        DenseVector model = new DenseVector(modelOfU.dim);
        DenseVector oldModel = new DenseVector(model.dim);

        long totalBegin = System.currentTimeMillis();

        for (int i = 0; ; i ++) {
            long startTrain = System.currentTimeMillis();
            //TODO StepSize tuning:  c/k(k=0,1,2...) or backtracking line search

            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                threadPool.execute(new executeRunnable(ThreadTrainCorpus.get(threadID),
                        modelOfU, modelOfV, lambda, corpus.size(), threadID));
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
            globalModelOfU.allDividedBy(threadNum);
            globalModelOfV.allDividedBy(threadNum);
            System.arraycopy(globalModelOfU.values, 0, modelOfU.values, 0, modelOfU.dim);
            System.arraycopy(globalModelOfV.values, 0, modelOfV.values, 0, modelOfV.dim);


            for(int j = 0; j < model.dim; j++){
                model.values[j] = modelOfU.values[j] - modelOfV.values[j];
            }
            long trainTime = System.currentTimeMillis() - startTrain;
            System.out.println("trainTime " + trainTime + " ");

            testAndSummary(trainCorpus, testCorpus, model, lambda);

            if(converge(oldModel, model)){
                if(earlyStop)
                    break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, oldModel.values.length);
            Arrays.fill(globalModelOfU.values, 0);
            Arrays.fill(globalModelOfV.values, 0);
            System.out.println("totaltime " + (System.currentTimeMillis() - totalBegin) );
            long nowCost = System.currentTimeMillis() - start;
            if(nowCost > maxTimeLimit) {
                break;
                //break;
            }
        }
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelGD.LogisticRegressionNesterovMomentum threadID FeatureDim train_path lambda learningRate [trainRatio]");
        threadNum = Integer.parseInt(argv[0]);
        int dimension = Integer.parseInt(argv[1]);
        String path = argv[2];
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dimension);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        lambda = Double.parseDouble(argv[3]);
        eta = Double.parseDouble(argv[4]);
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
        System.out.println("FeatureDimension " + dimension);
        System.out.println("LearningRate " + eta);
        System.out.println("File Path " + path);
        System.out.println("Lambda " + lambda);
        System.out.println("TrainRatio " + trainRatio);
        System.out.println("TimeLimit " + maxTimeLimit);
        System.out.println("EarlyStop " + earlyStop);
        LogisticRegressionNesterovMomentum lr = new LogisticRegressionNesterovMomentum();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        DenseVector modelOfU = new DenseVector(dimension);
        DenseVector modelOfV = new DenseVector(dimension);
        start = System.currentTimeMillis();
        lr.train(corpus, modelOfU, modelOfV);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
}
