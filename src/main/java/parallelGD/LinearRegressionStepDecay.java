package parallelGD;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;

import java.util.List;

public class LinearRegressionStepDecay extends LinearRegression{

    double decayRate = 0.7;
    int dacayIteration = 20;
    static double learningRate = 0.01;

    public void setNewLearningRate(){
        if(iteration % dacayIteration == 0){
            learningRate *= decayRate;
        }
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelGD.LinearRegressionExpDecay threadNum dim train_path learningRate [trainRatio]");
        threadNum = Integer.parseInt(argv[0]);
        int dim = Integer.parseInt(argv[1]);
        String path = argv[2];
        learningRate = Double.parseDouble(argv[3]);
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
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
        System.out.println("FeatureDimension " + dim);
        System.out.println("LearningRate " + learningRate);
        System.out.println("File Path " + path);
        System.out.println("TrainRatio " + trainRatio);
        System.out.println("TimeLimit " + maxTimeLimit);
        System.out.println("EarlyStop " + earlyStop);
        LinearRegressionStepDecay linear = new LinearRegressionStepDecay();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        DenseVector model = new DenseVector(dim);
        start = System.currentTimeMillis();
        linear.train(corpus, model);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
}
