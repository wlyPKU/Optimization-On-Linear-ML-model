
package parallelGD;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by 王羚宇 on 2016/7/20.
 */
public class LassoAdagrad extends model.Lasso{
    private static long start;

    private DenseVector globalModelOfU;
    private DenseVector globalModelOfV;
    private static int threadNum;
    private static double lambda = 0.1;
    private static double trainRatio = 0.5;

    static double learningRate = 0.05;
    int iteration = 1;

    double [][]G2ofU;
    double [][]G2ofV;
    //Guide value: 1e-8
    double epsilon = 1e-8;

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
            localList = list;
            localModelOfU = new DenseVector(modelOfU.dim);
            localModelOfV = new DenseVector(modelOfV.dim);
            System.arraycopy(modelOfU.values, 0, localModelOfU.values, 0, modelOfU.dim);
            System.arraycopy(modelOfV.values, 0, localModelOfV.values, 0, modelOfU.dim);
            this.lambda = lambda;
            this.globalCorpusSize = globalCorpusSize;
            this.threadID = threadID;

        }
        public void run() {
            sgdOneEpoch(localList, localModelOfU, localModelOfV, learningRate, lambda);
            globalModelOfU.plusDense(localModelOfU);
            globalModelOfV.plusDense(localModelOfV);
        }
        private void sgdOneEpoch(List<LabeledData> list, DenseVector modelOfU,
                                 DenseVector modelOfV, double lr, double lambda) {
            double modelPenalty = - lambda / globalCorpusSize;
            for (LabeledData labeledData: list) {
                double scala = labeledData.label - modelOfU.dot(labeledData.data)
                        + modelOfV.dot(labeledData.data);
                for(int i = 0; i < labeledData.data.indices.length; i++){
                    double gradient;
                    if(labeledData.data.values != null){
                        gradient = scala * labeledData.data.values[i] + modelPenalty;
                    }else{
                        gradient = scala + modelPenalty;
                    }
                    G2ofU[threadID][labeledData.data.indices[i]] += gradient * gradient;
                    modelOfU.values[labeledData.data.indices[i]] += lr * gradient /
                            Math.sqrt(G2ofU[threadID][labeledData.data.indices[i]] + epsilon);
                }
                modelOfU.positiveOrZero(labeledData.data);

                scala = labeledData.label - modelOfU.dot(labeledData.data)
                        + modelOfV.dot(labeledData.data);
                for(int i = 0; i < labeledData.data.indices.length; i++){
                    double gradient;
                    if(labeledData.data.values != null){
                        gradient = -scala * labeledData.data.values[i] + modelPenalty;
                    }else{
                        gradient = -scala + modelPenalty;
                    }
                    G2ofV[threadID][labeledData.data.indices[i]] += gradient * gradient;
                    modelOfV.values[labeledData.data.indices[i]] += lr * gradient
                            / Math.sqrt(G2ofV[threadID][labeledData.data.indices[i]] + epsilon);
                }
                modelOfV.positiveOrZero(labeledData.data);
            }
        }
    }

    public void train(List<LabeledData> corpus, DenseVector modelOfU,
                      DenseVector modelOfV) {
        Collections.shuffle(corpus);
        List<List<LabeledData>> ThreadTrainCorpus = new ArrayList<List<LabeledData>>();
        int size = corpus.size();
        int end = (int) (size * trainRatio);
        List<LabeledData> trainCorpus = corpus.subList(0, end);
        List<LabeledData> testCorpus = corpus.subList(end, size);
        for(int threadID = 0; threadID < threadNum; threadID++){
            int from = end * threadID / threadNum;
            int to = end * (threadID + 1) / threadNum;
            List<LabeledData> threadCorpus = corpus.subList(from, to);
            ThreadTrainCorpus.add(threadCorpus);
        }
        DenseVector model = new DenseVector(modelOfU.dim);
        DenseVector oldModel = new DenseVector(model.dim);

        globalModelOfU = new DenseVector(modelOfU.dim);
        globalModelOfV = new DenseVector(modelOfV.dim);

        G2ofU = new double[threadNum][modelOfU.dim];
        G2ofV = new double[threadNum][modelOfV.dim];

        for(int i = 0; i < threadNum; i++){
            for(int j = 0; j < model.dim; j++){
                G2ofV[i][j] = 0;
                G2ofU[i][j] = 0;
            }
        }


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
            System.out.println("totaltime " + (System.currentTimeMillis() - totalBegin) );
            if(converge(oldModel, model)){
                if(earlyStop)
                    break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, oldModel.values.length);
            Arrays.fill(globalModelOfU.values, 0);
            Arrays.fill(globalModelOfV.values, 0);
            long nowCost = System.currentTimeMillis() - start;
            if(nowCost > maxTimeLimit) {
                break;
                //break;
            }
            iteration++;
        }
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelGD.LassoAdagrad threadNum dim train_path lambda learningRate [trainRatio]");
        threadNum = Integer.parseInt(argv[0]);
        int dim = Integer.parseInt(argv[1]);
        String path = argv[2];
        lambda = Double.parseDouble(argv[3]);
        learningRate = Double.parseDouble(argv[4]);
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        if(argv.length >= 6){
            trainRatio = Double.parseDouble(argv[5]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }

        LassoAdagrad lasso = new LassoAdagrad();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        DenseVector modelOfU = new DenseVector(dim);
        DenseVector modelOfV = new DenseVector(dim);
        start = System.currentTimeMillis();
        lasso.train(corpus, modelOfU, modelOfV);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
}
