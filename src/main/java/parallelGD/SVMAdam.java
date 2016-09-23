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
 * Created by WLY on 2016/9/4.
 */
public class SVMAdam extends model.SVM{
    public DenseVector globalModel;
    public static double trainRatio = 0.5;
    public static int threadNum;
    public static double lambda = 0.1;

    public int iteration = 1;
    private static double learningRate = 0.001;
    double [][]mt;
    double [][]vt;
    double epsilon = 1e-6;
    double beta1 = 0.9;
    double beta2 = 0.999;

    double pow_beta1_t = beta1;
    double pow_beta2_t = beta2;


    public void setNewLearningRate(){
    }

    public class executeRunnable implements Runnable
    {
        List<LabeledData> localList;
        DenseVector localModel;
        double lambda;
        int globalCorpusSize;
        int threadID;
        public executeRunnable(List<LabeledData> list, DenseVector model, double lambda, int globalCorpusSize, int threadID){
            localList = list;
            localModel = new DenseVector(model.dim);
            System.arraycopy(model.values, 0, localModel.values, 0, model.dim);
            this.lambda = lambda;
            this.globalCorpusSize = globalCorpusSize;
            this.threadID = threadID;
        }
        public void run() {
            sgdOneEpoch(localList, localModel, lambda, globalCorpusSize);
            globalModel.plusDense(localModel);
        }
        public void sgdOneEpoch(List<LabeledData> list, DenseVector model, double lambda, double globalCorpusSize) {
            double modelPenalty = -2 * lambda / globalCorpusSize;
            for (LabeledData labeledData : list) {
                //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf Pg 3.
                /* model pennalty */
                //model.value[i] -= model.value[i] * 2 * lr * lambda / N;

                double dotProd = model.dot(labeledData.data);
                for(int i = 0; i < labeledData.data.indices.length; i++){
                    if(1 - dotProd * labeledData.label > 0) {
                        double gradient = modelPenalty * model.values[labeledData.data.indices[i]];
                        if(labeledData.data.values != null){
                            gradient += labeledData.label * labeledData.data.values[i];
                        }else{
                            gradient += labeledData.label;
                        }
                        mt[threadID][labeledData.data.indices[i]] = mt[threadID][labeledData.data.indices[i]] * beta1 +
                                (1 - beta1) * gradient;
                        vt[threadID][labeledData.data.indices[i]] = vt[threadID][labeledData.data.indices[i]] * beta2 +
                                (1 - beta2) * gradient * gradient;
                        double mt_hat = mt[threadID][labeledData.data.indices[i]] / (1 - pow_beta1_t);
                        double vt_hat = vt[threadID][labeledData.data.indices[i]] / (1 - pow_beta2_t);

                        model.values[labeledData.data.indices[i]] += learningRate * mt_hat/ (epsilon + Math.sqrt(vt_hat));
                    }
                }
            }
        }
    }

    public void train(List<LabeledData> corpus, DenseVector model) {
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

        mt = new double[threadNum][model.dim];
        vt = new double[threadNum][model.dim];

        for(int i = 0; i < threadNum; i++){
            for(int j = 0; j < model.dim; j++) {
                mt[i][j] = 0;
                vt[i][j] = 0;
            }
        }

        DenseVector oldModel = new DenseVector(model.values.length);
        globalModel = new DenseVector(model.dim);

        long totalBegin = System.currentTimeMillis();

        for (int i = 0; i < 200; i ++) {
            long startTrain = System.currentTimeMillis();
            System.out.println("learning rate " + learningRate);

            //StepSize tuning:  c/k(k=0,1,2...) or backtracking line search
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                threadPool.execute(new executeRunnable(ThreadTrainCorpus.get(threadID),
                        model, lambda, trainCorpus.size(), threadID));
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
            globalModel.allDividedBy(threadNum);
            System.arraycopy(globalModel.values, 0, model.values, 0, model.dim);
            Arrays.fill(globalModel.values, 0);

            long trainTime = System.currentTimeMillis() - startTrain;
            System.out.println("trainTime " + trainTime + " ");
            testAndSummary(trainCorpus, testCorpus, model, lambda);

            if(converge(oldModel, model)){
                //break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, oldModel.values.length);
            System.out.println("totaltime " + (System.currentTimeMillis() - totalBegin) );

            iteration++;
            pow_beta1_t *= beta1;
            pow_beta2_t *= beta2;
            setNewLearningRate();
        }
    }


    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelGD.SVMAdam threadNum dim train_path lambda learningRate [trainRatio]");
        threadNum = Integer.parseInt(argv[0]);
        int dim = Integer.parseInt(argv[1]);
        String path = argv[2];
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        lambda = Double.parseDouble(argv[3]);
        learningRate = Double.parseDouble(argv[4]);
        if(argv.length >= 6){
            trainRatio = Double.parseDouble(argv[5]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }

        SVMAdam svm = new SVMAdam();
        DenseVector model = new DenseVector(dim);
        long start = System.currentTimeMillis();
        svm.train(corpus, model);

        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
}
