package model;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;
import java.util.List;

/**
 * Created by 王羚宇 on 2016/8/7.
 */
public class LinearRegression {
    public static boolean earlyStop = true;
    public static double maxTimeLimit = 200000;
    public static double stopDelta = 0.00001;
    public static int modelType = 0;
    public static int maxIteration = 100;
    public static boolean rhoFixed = true;

    public static boolean doNormalize =true;

    public boolean testAndSummary(List<LabeledData>trainCorpus, List<LabeledData> testCorpus,
                                DenseVector model){
        long startTest = System.currentTimeMillis();
        double loss = test(trainCorpus, model);
        double accuracy = test(testCorpus, model);
        long testTime = System.currentTimeMillis() - startTest;
        System.out.println("[Information]TrainLoss=" + loss + " TestLoss=" + accuracy +
                " TestTime=" + testTime);
        System.out.println("[Information]AverageTrainLoss=" + loss / trainCorpus.size() + " AverageTestLoss=" + accuracy / testCorpus.size());
        double []trainAccuracy = Utils.LinearAccuracy(trainCorpus, model);
        double []testAccuracy = Utils.LinearAccuracy(testCorpus, model);
        //System.out.println("trainAccuracy:");
        //Utils.printAccuracy(trainAccuracy);
        //System.out.println("testAccuracy:");
        //Utils.printAccuracy(testAccuracy);
        return loss > 1e100 || Double.isInfinite(loss) || Double.isNaN(loss);
    }

    public boolean testAndSummary(List<LabeledData>trainCorpus, List<LabeledData> testCorpus,
                                  DenseVector model, boolean verbose){
        long startTest = System.currentTimeMillis();
        double loss = test(trainCorpus, model);
        double accuracy = test(testCorpus, model);
        long testTime = System.currentTimeMillis() - startTest;
        if(verbose) {
            System.out.println("[Information]TrainLoss=" + loss + " TestLoss=" + accuracy +
                    " TestTime=" + testTime);
            System.out.println("[Information]AverageTrainLoss=" + loss / trainCorpus.size() + " AverageTestLoss=" + accuracy / testCorpus.size());
            double[] trainAccuracy = Utils.LinearAccuracy(trainCorpus, model);
            double[] testAccuracy = Utils.LinearAccuracy(testCorpus, model);
            //System.out.println("trainAccuracy:");
            //Utils.printAccuracy(trainAccuracy);
            //System.out.println("testAccuracy:");
            //Utils.printAccuracy(testAccuracy);
        }
        return loss > 1e100 || Double.isInfinite(loss) || Double.isNaN(loss);
    }

    public double test(List<LabeledData> list, DenseVector model) {
        double residual = 0;
        for (LabeledData labeledData : list) {
            double dot_prod = model.dot(labeledData.data);
            residual += 0.5 * Math.pow(labeledData.label - dot_prod, 2);
        }
        return residual;
    }
    public boolean converge(DenseVector oldModel, DenseVector newModel, List<LabeledData> data){
        double delta = 0;
        for(int i = 0; i < oldModel.values.length; i++){
            delta += Math.pow(oldModel.values[i] - newModel.values[i], 2);
        }
        System.out.println("[Information]LossChanged " + (test(data, oldModel)
                - test(data, newModel)));
        System.out.println("[Information]LossAbsoluteChanged " + (Math.abs(test(data, oldModel)
                - test(data, newModel))));
        System.out.println("[Information]ParameterChanged " + delta);
        System.out.println("[Information]AverageParameterChanged " + Math.sqrt(delta) / oldModel.values.length);

        return delta < stopDelta;
    }
    public boolean converge(DenseVector oldModel, DenseVector newModel, List<LabeledData> data, boolean verbose){
        double delta = 0;
        for(int i = 0; i < oldModel.values.length; i++){
            delta += Math.pow(oldModel.values[i] - newModel.values[i], 2);
        }
        if(verbose) {
            System.out.println("[Information]LossChanged " + (test(data, oldModel)
                    - test(data, newModel)));
            System.out.println("[Information]LossAbsoluteChanged " + (Math.abs(test(data, oldModel)
                    - test(data, newModel))));
            System.out.println("[Information]ParameterChanged " + delta);
            System.out.println("[Information]AverageParameterChanged " + Math.sqrt(delta) / oldModel.values.length);
        }
        return delta < stopDelta;
    }
}
