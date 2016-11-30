package model;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;
import org.apache.log4j.Logger;

import java.util.List;

/**
 * Created by 王羚宇 on 2016/8/7.
 */

public class Lasso {
    public static boolean earlyStop = true;
    public static double maxTimeLimit = 200000;
    public static double stopDelta = 0.00001;
    public static int maxIteration = 100;
    public static int modelType = 0;
    public static boolean rhoFixed = true;

    public boolean testAndSummary(List<LabeledData>trainCorpus, List<LabeledData> testCorpus,
                                DenseVector x, double lambda){
        long startTest = System.currentTimeMillis();
        double trainLoss = lassoLoss(trainCorpus, x, lambda);
        double testLoss = lassoLoss(testCorpus, x, lambda);
        double trainResidual = test(trainCorpus, x);
        double testResidual = test(testCorpus, x);
        long testTime = System.currentTimeMillis() - startTest;
        System.out.println("[Information]TrainLoss=" + trainLoss + " TestLoss=" + testLoss +
                " TrainResidual=" + trainResidual + " TestResidual=" + testResidual +
                " TestTime=" + testTime);
        System.out.println("[Information]AverageTrainLoss=" + trainLoss / trainCorpus.size() + " AverageTestLoss=" + testLoss / testCorpus.size() +
                " AverageTrainResidual=" + trainResidual / trainCorpus.size() + " AverageTestResidual=" + testResidual / testCorpus.size());
        double []trainAccuracy = Utils.LinearAccuracy(trainCorpus, x);
        double []testAccuracy = Utils.LinearAccuracy(testCorpus, x);
        //System.out.println("trainAccuracy:");
        //Utils.printAccuracy(trainAccuracy);
        //System.out.println("testAccuracy:");
        //Utils.printAccuracy(testAccuracy);
        return (trainResidual > 2e100) || Double.isInfinite(trainLoss) || Double.isNaN(trainLoss);
    }
    @SuppressWarnings("unused")
    public double lassoLoss(List<LabeledData> list, DenseVector model_x, DenseVector model_z, double lambda) {
        double loss = 0.0;
        for (LabeledData labeledData: list) {
            double predictValue = model_x.dot(labeledData.data);
            loss += 1.0 / 2.0 * Math.pow(labeledData.label - predictValue, 2);
        }
        for(Double v: model_z.values){
            loss += lambda * (v > 0? v : -v);
        }
        return loss;
    }
    public double test(List<LabeledData> list, DenseVector model) {
        double residual = 0;
        for (LabeledData labeledData : list) {
            double dot_prod = model.dot(labeledData.data);
            residual += 0.5 * Math.pow(labeledData.label - dot_prod, 2);
        }
        return residual;
    }
    private double lassoLoss(List<LabeledData> list, DenseVector model, double lambda) {
        double loss = 0.0;
        for (LabeledData labeledData: list) {
            double predictValue = model.dot(labeledData.data);
            loss += 1.0 / 2.0 * Math.pow(labeledData.label - predictValue, 2);
        }
        for(Double v: model.values){
            loss += lambda * (v > 0? v : -v);
        }
        return loss;
    }
    public boolean converge(DenseVector oldModel, DenseVector newModel){
        double delta = 0;
        for(int i = 0; i < oldModel.values.length; i++){
            delta += Math.pow(oldModel.values[i] - newModel.values[i], 2);
        }
        System.out.println("[Information]ParameterChanged " + delta);
        System.out.println("[Information]AverageParameterChanged " + Math.sqrt(delta) / oldModel.values.length);
        return delta < stopDelta;
    }
}
